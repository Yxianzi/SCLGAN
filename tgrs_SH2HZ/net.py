import torch
import torch.nn as nn
import torch.nn.functional as F
from BFIN import bfin

from einops import rearrange, repeat
try:
    from thop import profile
except ImportError:
    profile = None

#class MambaFeature(nn.Module):
    #def __init__(self, n_bands, patch_size, encoder_embed_dim=144, emb_dropout=0., out_dim=256):
        #super(MambaFeature, self).__init__()

        #self.input_bands = n_bands
        #self.out_dim = out_dim
        #self.patch_size = patch_size

        #self.conv1_1 = nn.Sequential(
            #nn.Conv2d(n_bands, 32, 3, 1, 1),
            #nn.BatchNorm2d(32),
            #nn.ReLU(),  # No effect on order
        #)
        #self.conv1_2 = nn.Sequential(
            #nn.Conv2d(32, 64, 3, 1, 1),
            #nn.BatchNorm2d(64),
            #nn.ReLU(),  # No effect on order
        #)
        #self.conv2 = nn.Conv2d(64, 64, 1)
        #.conv3 = nn.Conv2d(64, 32, 1)
        #self.encoder_embedding = nn.Linear((patch_size * 1) ** 2, self.patch_size ** 2)
        #self.encoder_pos_embed = nn.Parameter(torch.randn(1, self.patch_size ** 2 + 1 + 2, encoder_embed_dim))
        #self.dropout = nn.Dropout(emb_dropout)
        #self.mamba = Mamba(encoder_embed_dim)
        #self.relu = nn.ReLU(4608)
        #self.bn = nn.BatchNorm2d(32)

    # forward(self, x):
        #x = self.conv1_1(x)
        #x = self.conv1_2(x)

        #x = x.flatten(2)
        #x = self.encoder_embedding(x)
        # x = torch.einsum('nld->ndl', x)
        # x += self.encoder_pos_embed[:, :1]
        # x = self.dropout(x)
        # x = torch.einsum('nld->ndl', x)

        #b, c, h = x.shape
        #x = self.conv2(x.reshape(b, c, 12, 12)).reshape(b, c, h)
        #x = self.mamba(x) + x
        #x = self.relu(self.bn(self.conv3(x.reshape(b, c, 12, 12)))).reshape(b, 4608, 1, 1)

        #return x

class F_encoder(nn.Module):
    def __init__(self, batch,n_band=198, patch_size=3,num_class=3,num_tokens=49):
        super(F_encoder, self).__init__()
        self.n_outputs = 288
        self.multi_level_feature_selector = CTrans_encoder(n_band,patch_size,num_class,batch,num_tokens)

        self.fc1 = nn.Linear(288, num_class)
        self.fc2 = nn.Linear(288, 1)

        self.head1 = nn.Sequential(
            nn.Linear(288, 128),
            # nn.ReLU(inplace=True),
            # nn.Linear(288, 128)
        )
        self.head2 = nn.Sequential(
            nn.Linear(288, 128),
            # nn.ReLU(inplace=True),
            # nn.Linear(288, 128)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y,false = 'train'):
        if false == 'test':
            output_y,low_mask_t, high_mask_t, loss_t  = self.multi_level_feature_selector(x, y, false='test')

            fea_y = F.normalize(self.head1(output_y), dim=1)

            cl_tgr = self.fc1(output_y)

            out_y = self.fc2(output_y)
            out_y = self.sigmoid(out_y)

            return output_y, fea_y, cl_tgr, out_y

        else:
            output_s, low_mask_s, high_mask_s, loss_s, output_y, low_mask_y, high_mask_y, loss_y = self.multi_level_feature_selector(
                x, y)

            fea_s = F.normalize(self.head1(output_s), dim=1)

            fea_s1 = F.normalize(self.head2(output_s), dim=1)

            cl_src = self.fc1(output_s)

            out_s = self.fc2(output_s)
            out_s = self.sigmoid(out_s)

            fea_y = F.normalize(self.head1(output_y), dim=1)
            fea_y1 = F.normalize(self.head2(output_y), dim=1)

            cl_tgr = self.fc1(output_y)

            out_y = self.fc2(output_y)
            out_y = self.sigmoid(out_y)

            return output_s, fea_s, cl_src, out_s,  fea_s1, loss_s, output_y, fea_y, cl_tgr, out_y, fea_y1, loss_y

    def get_embedding(self, x, y):
        out, _, _, _ = self.forward(x, y,false='test')
        return out

class DSAN1(nn.Module):
    def __init__(self, n_band=198, patch_size=3,num_class=3):
        super(DSAN1, self).__init__()
        self.n_outputs = 288
        self.feature_layers = CTrans_encoder(n_band,patch_size,num_class)

        self.fc1 = nn.Linear(288, num_class)
        self.fc2 = nn.Linear(288, 1)

        self.head1 = nn.Sequential(
            nn.Linear(288, 64),
            # nn.ReLU(inplace=True),
            # nn.Linear(288, 128)
        )
        self.head2 = nn.Sequential(
            nn.Linear(288, 64),
            # nn.ReLU(inplace=True),
            # nn.Linear(288, 128)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        features = self.feature_layers(x)

        x1 = F.normalize(self.head1(features), dim=1)
        x2 = F.normalize(self.head2(features), dim=1)

        fea = self.fc1(features)
        output = self.fc2(features)
        output = self.sigmoid(output)

        return features,x1,x2,fea, output

    def get_embedding(self, x):
        out, _, _, _, _ = self.forward(x)
        return out

class DSAN2(nn.Module):
    def __init__(self, n_band=198, patch_size=3,num_class=3):
        super(DSAN2, self).__init__()
        self.n_outputs = 152
        self.feature_layers = DCRN(n_band,patch_size,num_class)

        self.fc1 = nn.Linear(self.n_outputs, num_class)
        self.fc2 = nn.Linear(self.n_outputs, 1)

        self.head1 = nn.Sequential(
            nn.Linear(self.n_outputs, 128),
            # nn.ReLU(inplace=True),
            # nn.Linear(288, 128)
        )
        self.head2 = nn.Sequential(
            nn.Linear(self.n_outputs, 128),
            # nn.ReLU(inplace=True),
            # nn.Linear(288, 128)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        features = self.feature_layers(x)

        x1 = F.normalize(self.head1(features), dim=1)
        x2 = F.normalize(self.head2(features), dim=1)

        fea = self.fc1(features)
        output = self.fc2(features)
        output = self.sigmoid(output)

        return features, x1, x2, fea, output

    def get_embedding(self, x):
        out, _, _, _, _ = self.forward(x)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(MAA(dim, heads, dim_head, dropout))

    def forward(self, x, memories):
        for attn in self.layers:
            x = attn(x, memories=memories) + x
        return x

class MAA(nn.Module):
    def __init__(self, dim, heads=8, dim_head=288, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_memory = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, memories):
        x = self.norm(x)
        x_kv = x

        q, k, v = (self.to_q(x), *self.to_kv(x_kv).chunk(2, dim=-1))
        memories = self.to_memory(memories)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        memories = rearrange(memories, 'b n (h d) -> b h n d', h=self.heads)

        k = torch.cat((k, memories), dim=2)
        v = torch.cat((v, memories), dim=2)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.1):
        super().__init__()
        self.layernorm = nn.LayerNorm(dim)
        self.nn1 = nn.Linear(dim, hidden_dim)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.nn2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x = self.layernorm(x)
        x = self.nn1(x)
        x = self.gelu(x)
        x = self.drop(x)
        x = self.nn2(x)
        x = self.drop(x)
        return x

class CTrans_encoder(nn.Module):

    def __init__(self, input_channels, patch_size, n_classes,batch,num_tokens=1,input_dim=288, dropout=0.2,
                 emb_dropout=0.1, output_dim=128, model_dim=288, num_heads=8,dim_head=8, num_layers=1,dim=288, depth = 4,heads = 8,
                 mlp_dim = 512):
        super(CTrans_encoder, self).__init__()
        self.kernel_dim = 1
        self.feature_dim = input_channels
        self.sz = patch_size
        if patch_size == 7:
            self.mp = nn.MaxPool2d(4)
            self.dim1 = 128
            self.dim2 = 32
            self.input_dim = 160
        elif patch_size == 1:
            self.mp = nn.MaxPool2d(1)
            self.dim1 = 128
            self.dim2 =32
            self.input_dim = 160
        elif patch_size == 3:
            self.mp = nn.MaxPool2d(2)
            self.dim1 = 128
            self.dim2 = 32
            self.input_dim = 160
        elif patch_size == 11:
            self.mp = nn.MaxPool2d(6)
            self.dim1 = 128
            self.dim2 = 32
            self.input_dim = 160
        elif patch_size == 5:
            self.mp = nn.MaxPool2d(3)
            self.dim1 = 128
            self.dim2 = 32
            self.input_dim = 160
        elif patch_size == 13:
            self.mp = nn.MaxPool2d(7)
            self.dim1 = 128
            self.dim2 = 32
            self.input_dim = 160
        elif patch_size == 9:
            self.mp = nn.MaxPool2d(5)
            self.dim1 = 128
            self.dim2 = 32
            self.input_dim = 160
        # Convolution Layer 1 kernel_size = (1, 1, 7), stride = (1, 1, 2), output channels = 24
        self.conv1 = nn.Conv3d(1, 24, kernel_size=(7, 1, 1), stride=(2, 1, 1), bias=True)
        self.bn1 = nn.BatchNorm3d(24)
        self.activation1 = nn.ReLU()

        # Residual block 1
        self.conv2 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0),
                               bias=True)  # padding_mode='replicate',
        self.bn2 = nn.BatchNorm3d(24)
        self.activation2 = nn.ReLU()
        self.conv3 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0),
                               bias=True)  # padding_mode='replicate',
        self.bn3 = nn.BatchNorm3d(24)
        self.activation3 = nn.ReLU()
        # Finish

        # Convolution Layer 2 kernel_size = (1, 1, (self.feature_dim - 6) // 2), output channels = 128
        self.conv4 = nn.Conv3d(24, self.dim1, kernel_size=(((self.feature_dim - 7) // 2 + 1), 1, 1), bias=True)
        self.bn4 = nn.BatchNorm3d(self.dim1)
        self.activation4 = nn.ReLU()

        # Convolution layer for spatial information
        self.conv5 = nn.Conv3d(1, 24, (self.feature_dim, 1, 1))
        self.bn5 = nn.BatchNorm3d(24)
        self.activation5 = nn.ReLU()

        # Residual block 2
        self.conv6 = nn.Conv3d(24, 24, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                               bias=True)  # padding_mode='replicate',
        self.bn6 = nn.BatchNorm3d(24)
        self.activation6 = nn.ReLU()
        self.conv7 = nn.Conv3d(24, self.dim2, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                               bias=True)  # padding_mode='replicate',
        self.bn7 = nn.BatchNorm3d(self.dim2)
        self.activation7 = nn.ReLU()
        self.conv8 = nn.Conv3d(24, self.dim2, kernel_size=1)
        # Finish

        # Combination shape
        # self.inter_size = 128 + 24
        self.inter_size = self.dim1 + self.dim2

        # Residual block 3
        self.conv9 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                               bias=True)  # padding_mode='replicate',
        self.bn9 = nn.BatchNorm3d(self.inter_size)
        self.activation9 = nn.ReLU()
        self.conv10 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                                bias=True)  # padding_mode='replicate',
        self.bn10 = nn.BatchNorm3d(self.inter_size)
        self.activation10 = nn.ReLU()

        # attention
        self.ca = ChannelAttention(self.inter_size)
        self.sa = SpatialAttention()

        self.avgpool = nn.AvgPool3d((1, self.sz, self.sz))
        self.fc1 = nn.Linear(in_features=self.inter_size, out_features=n_classes)
        self.bfin = bfin(cnn_dim=self.input_dim, former_dim=self.input_dim,num_dim=288, batch=batch,n_classes=n_classes)

        # attention
        self.nn = nn.Linear(num_tokens, self.input_dim)
        torch.nn.init.xavier_uniform_(self.nn.weight)
        self.dropout = nn.Dropout(dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.input_dim))
        self.pos_embedding = nn.Parameter(torch.empty(1, num_tokens + 1, self.input_dim))
        torch.nn.init.normal_(self.pos_embedding, std=.001)
        self.transformer = Transformer(self.input_dim, depth, heads, dim_head, mlp_dim, emb_dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.input_dim)
        )
        self.drop = nn.Dropout(emb_dropout)

        # parameters initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x,y,false = 'train'):
        if false == 'test':
            in_size_y = y.size(0)  # 32
            y = y.unsqueeze(1)  # (64,1,100,9,9)
            # Convolution layer 1
            y1 = self.conv1(y)
            y1 = self.activation1(self.bn1(y1))
            # Residual layer 1
            residual_y = y1
            y1 = self.conv2(y1)
            y1 = self.activation2(self.bn2(y1))
            y1 = self.conv3(y1)
            y1 = residual_y + y1  # (32,24,21,7,7)
            y1 = self.activation3(self.bn3(y1))

            # Convolution layer to combine rest
            y1 = self.conv4(y1)  # (32,128,1,7,7)
            y1 = self.activation4(self.bn4(y1))
            y1 = y1.reshape(y1.size(0), y1.size(1), y1.size(3), y1.size(4))  # (32,128,7,7)

            y2 = self.conv5(y)  # (32,24,1,7,7)
            y2 = self.activation5(self.bn5(y2))

            # Residual layer 2
            residual_y2 = y2
            residual_y22 = self.conv8(residual_y2)  # (32,24,1,7,7)
            y2 = self.conv6(y2)  # (32,24,1,7,7)
            y2 = self.activation6(self.bn6(y2))
            y2 = self.conv7(y2)  # (32,24,1,7,7)
            y2 = residual_y22 + y2

            y2 = self.activation7(self.bn7(y2))
            y2 = y2.reshape(y2.size(0), y2.size(1), y2.size(3), y2.size(4))  # (32,24,7,7)

            # concat spatial and spectral information
            y = torch.cat((y1, y2), 1)  # (32,152,7,7)
            ###################
            # attention map
            ###################
            ###################
            # attention map
            ###################
            y = self.ca(y) * y
            y = self.sa(y) * y

            y_c1 = self.avgpool(y)

            y_c = y_c1.view(y_c1.shape[0], -1)  # (288)

            avgpool_token_y = self.avgpool(y)
            max_token_y = self.mp(y)  # 32, 128, 1, 1

            memory1_y = avgpool_token_y.view(in_size_y, -1)
            memory2_y = max_token_y.view(in_size_y, -1)
            memory_y = torch.cat((memory1_y.unsqueeze(1), memory2_y.unsqueeze(1)), dim=1)  # 32 2 128
            memory_y = self.drop(memory_y)

            out3 = rearrange(y_c1, 'b c h w -> b (h w) c')  # 32 9 128
            cls_token = repeat(self.cls_token, '1 n d -> b n d', b=out3.shape[0])
            y3 = torch.cat((cls_token, out3), dim=1)
            y3 += self.pos_embedding
            y3 = self.dropout(y3)
            y3 = self.transformer(y3, memory_y)  # 32,2,128
            token_y = y3[:, 0]
            y_f = self.mlp_head(token_y)
            yy_cf, low_mask_t, high_mask_t, loss_t = self.bfin(y_c, y_f, y_c, y_f, false='test')  # 32,64

            return yy_cf, low_mask_t, high_mask_t, loss_t

        else:
            in_size = x.size(0)  # 32

            x = x.unsqueeze(1)  # (64,1,100,9,9)
            # Convolution layer 1
            x1 = self.conv1(x)
            x1 = self.activation1(self.bn1(x1))
            # Residual layer 1
            residual = x1
            x1 = self.conv2(x1)
            x1 = self.activation2(self.bn2(x1))
            x1 = self.conv3(x1)
            x1 = residual + x1  # (32,24,21,7,7)
            x1 = self.activation3(self.bn3(x1))

            # Convolution layer to combine rest
            x1 = self.conv4(x1)  # (32,128,1,7,7)
            x1 = self.activation4(self.bn4(x1))
            x1 = x1.reshape(x1.size(0), x1.size(1), x1.size(3), x1.size(4))  # (32,128,7,7)

            x2 = self.conv5(x)  # (32,24,1,7,7)
            x2 = self.activation5(self.bn5(x2))

            # Residual layer 2
            residual = x2
            residual = self.conv8(residual)  # (32,24,1,7,7)
            x2 = self.conv6(x2)  # (32,24,1,7,7)
            x2 = self.activation6(self.bn6(x2))
            x2 = self.conv7(x2)  # (32,24,1,7,7)
            x2 = residual + x2

            x2 = self.activation7(self.bn7(x2))
            x2 = x2.reshape(x2.size(0), x2.size(1), x2.size(3), x2.size(4))  # (32,24,7,7)

            # concat spatial and spectral information
            x = torch.cat((x1, x2), 1)  # (32,152,7,7)
            ###################
            # attention map
            ###################
            ###################
            # attention map
            ###################
            x = self.ca(x) * x
            x = self.sa(x) * x

            x_c1 = self.avgpool(x)
            x_c = x_c1.view(x_c1.shape[0], -1)  # (288)

            avgpool_token = self.avgpool(x)
            max_token = self.mp(x)  # 32, 128, 1, 1

            memory1 = avgpool_token.view(in_size, -1)
            memory2 = max_token.view(in_size, -1)
            memory = torch.cat((memory1.unsqueeze(1), memory2.unsqueeze(1)), dim=1)  # 32 2 128
            memory = self.drop(memory)

            out3 = rearrange(x_c1, 'b c h w -> b (h w) c')  # 32 9 128
            cls_token = repeat(self.cls_token, '1 n d -> b n d', b=out3.shape[0])
            x3 = torch.cat((cls_token, out3), dim=1)
            x3 += self.pos_embedding
            x3 = self.dropout(x3)
            x3 = self.transformer(x3, memory)  # 32,2,128
            token_x = x3[:, 0]
            x_f = self.mlp_head(token_x)

            in_size_y = y.size(0)  # 32
            y = y.unsqueeze(1)  # (64,1,100,9,9)
            # Convolution layer 1
            y1 = self.conv1(y)
            y1 = self.activation1(self.bn1(y1))
            # Residual layer 1
            residual_y = y1
            y1 = self.conv2(y1)
            y1 = self.activation2(self.bn2(y1))
            y1 = self.conv3(y1)
            y1 = residual_y + y1  # (32,24,21,7,7)
            y1 = self.activation3(self.bn3(y1))

            # Convolution layer to combine rest
            y1 = self.conv4(y1)  # (32,128,1,7,7)
            y1 = self.activation4(self.bn4(y1))
            y1 = y1.reshape(y1.size(0), y1.size(1), y1.size(3), y1.size(4))  # (32,128,7,7)

            y2 = self.conv5(y)  # (32,24,1,7,7)
            y2 = self.activation5(self.bn5(y2))

            # Residual layer 2
            residual_y2 = y2
            residual_y22 = self.conv8(residual_y2)  # (32,24,1,7,7)
            y2 = self.conv6(y2)  # (32,24,1,7,7)
            y2 = self.activation6(self.bn6(y2))
            y2 = self.conv7(y2)  # (32,24,1,7,7)
            y2 = residual_y22 + y2

            y2 = self.activation7(self.bn7(y2))
            y2 = y2.reshape(y2.size(0), y2.size(1), y2.size(3), y2.size(4))  # (32,24,7,7)

            # concat spatial and spectral information
            y = torch.cat((y1, y2), 1)  # (32,152,7,7)
            ###################
            # attention map
            ###################
            ###################
            # attention map
            ###################
            y = self.ca(y) * y
            y = self.sa(y) * y

            y_c1 = self.avgpool(y)

            y_c = y_c1.view(y_c1.shape[0], -1)  # (288)

            avgpool_token_y = self.avgpool(y)
            max_token_y = self.mp(y)  # 32, 128, 1, 1

            memory1_y = avgpool_token_y.view(in_size_y, -1)
            memory2_y = max_token_y.view(in_size_y, -1)
            memory_y = torch.cat((memory1_y.unsqueeze(1), memory2_y.unsqueeze(1)), dim=1)  # 32 2 128
            memory_y = self.drop(memory_y)

            out3 = rearrange(y_c1, 'b c h w -> b (h w) c')  # 32 9 128
            cls_token = repeat(self.cls_token, '1 n d -> b n d', b=out3.shape[0])
            y3 = torch.cat((cls_token, out3), dim=1)
            y3 += self.pos_embedding
            y3 = self.dropout(y3)
            y3 = self.transformer(y3, memory_y)
            token_y = y3[:, 0]
            y_f = self.mlp_head(token_y)

            xx_cf, low_mask_s, high_mask_s, loss_s, yy_cf, low_mask_t, high_mask_t, loss_t = self.bfin(x_c, x_f, y_c,y_f)  # 32,64

            return xx_cf, low_mask_s, high_mask_s, loss_s, yy_cf, low_mask_t, high_mask_t, loss_t


class DCRN(nn.Module):

    def __init__(self, input_channels, patch_size, n_classes):
        super(DCRN, self).__init__()
        self.kernel_dim = 1
        self.feature_dim = input_channels
        self.sz = patch_size
        # Convolution Layer 1 kernel_size = (1, 1, 7), stride = (1, 1, 2), output channels = 24
        self.conv1 = nn.Conv3d(1, 24, kernel_size=(7, 1, 1), stride=(2, 1, 1), bias=True)
        self.bn1 = nn.BatchNorm3d(24)
        self.activation1 = nn.ReLU()

        # Residual block 1
        self.conv2 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0), bias=True)#padding_mode='replicate',
        self.bn2 = nn.BatchNorm3d(24)
        self.activation2 = nn.ReLU()
        self.conv3 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0),bias=True)# padding_mode='replicate',
        self.bn3 = nn.BatchNorm3d(24)
        self.activation3 = nn.ReLU()
        # Finish

        # Convolution Layer 2 kernel_size = (1, 1, (self.feature_dim - 6) // 2), output channels = 128
        self.conv4 = nn.Conv3d(24, 128, kernel_size=(((self.feature_dim - 7) // 2 + 1), 1, 1), bias=True)
        self.bn4 = nn.BatchNorm3d(128)
        self.activation4 = nn.ReLU()

        # Convolution layer for spatial information
        self.conv5 = nn.Conv3d(1, 24, (self.feature_dim, 1, 1))
        self.bn5 = nn.BatchNorm3d(24)
        self.activation5 = nn.ReLU()

        # Residual block 2
        self.conv6 = nn.Conv3d(24, 24, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=True)#padding_mode='replicate',
        self.bn6 = nn.BatchNorm3d(24)
        self.activation6 = nn.ReLU()
        self.conv7 = nn.Conv3d(24, 24, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=True)#padding_mode='replicate',
        self.bn7 = nn.BatchNorm3d(24)
        self.activation7 = nn.ReLU()
        self.conv8 = nn.Conv3d(24, 24, kernel_size=1)
        # Finish

        # Combination shape
        self.inter_size = 128 + 24



        # Residual block 3
        self.conv9 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=True)#padding_mode='replicate',
        self.bn9 = nn.BatchNorm3d(self.inter_size)
        self.activation9 = nn.ReLU()
        self.conv10 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),bias=True)#padding_mode='replicate',
        self.bn10 = nn.BatchNorm3d(self.inter_size)
        self.activation10 = nn.ReLU()

        # attention
        self.ca = ChannelAttention(self.inter_size)#self.inter_size
        self.sa = SpatialAttention()

        # Average pooling kernel_size = (5, 5, 1)
        self.avgpool = nn.AvgPool3d((1, self.sz, self.sz))

        # Fully connected Layer
        self.fc1 = nn.Linear(in_features=self.inter_size, out_features=n_classes)

        # parameters initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m,nn.Linear):
            #     torch.nn.init.kaiming_normal_(m.weight.data)
            #     m.bias.data = torch.ones(m.bias.data.size())



    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.xavier_uniform_(m.weight, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight, 1.0, 0.02)
            m.bias.data.zero_()
        elif classname.find('Linear') != -1:

            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data = torch.ones(m.bias.data.size())

    def forward(self, x):

        x = x.unsqueeze(1) # (64,1,100,9,9)
        # Convolution layer 1
        x1 = self.conv1(x)
        x1 = self.activation1(self.bn1(x1))
        # Residual layer 1
        residual = x1
        x1 = self.conv2(x1)
        x1 = self.activation2(self.bn2(x1))
        x1 = self.conv3(x1)
        x1 = residual + x1                  #(32,24,21,7,7)
        x1 = self.activation3(self.bn3(x1))

        # Convolution layer to combine rest
        x1 = self.conv4(x1)                 #(32,128,1,7,7)
        x1 = self.activation4(self.bn4(x1))
        x1 = x1.reshape(x1.size(0), x1.size(1), x1.size(3), x1.size(4)) #(32,128,7,7)

        ###########
        # attention model
        #BAM
        ###########
        # x1 = self.ca(x1) * x1


        ###########################
        #spatial

        x2 = self.conv5(x)                      #(32,24,1,7,7)
        x2 = self.activation5(self.bn5(x2))

        # Residual layer 2
        residual = x2
        residual = self.conv8(residual)     #(32,24,1,7,7)
        x2 = self.conv6(x2)                 #(32,24,1,7,7)
        x2 = self.activation6(self.bn6(x2))
        x2 = self.conv7(x2)                 #(32,24,1,7,7)
        x2 = residual + x2

        x2 = self.activation7(self.bn7(x2))
        x2 = x2.reshape(x2.size(0), x2.size(1), x2.size(3), x2.size(4)) #(32,24,7,7)

        ################
        #attention model
        ################
        # x2 = self.sa(x2) * x2
        ##SAM


        # concat spatial and spectral information
        # x1 = x1 * self.ca(x1)
        # x2 = x2 * self.sa(x2)

        x = torch.cat((x1, x2), 1)      #(32,152,7,7)


        ###################
        # attention map
        ###################
        x = self.ca(x) * x
        x = self.sa(x) * x

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)  # (288)



        #####################
        # attention map over
        #####################
        #CMMD


        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 4, 1, bias=False) #4-->16
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 4, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
