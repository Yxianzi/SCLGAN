import torch.nn.functional as F
import torch
import torch.nn as nn
from involution import involution
from CrosAttention import SA,SE


class Spectral_Weight(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(Spectral_Weight, self).__init__()
        self.f_inv_11 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, dilation, groups, bias)
        self.f_inv_12 = involution(in_channels, kernel_size, 1)
        self.bn_h = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X_h):
        x1 = self.f_inv_12(X_h)
        x2 = self.f_inv_11(x1)
        X_h = self.relu(self.bn_h(x2))
        return X_h
    
class Spatial_Weight(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(Spatial_Weight, self).__init__()
        self.Conv_weight = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn_h = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X_h):
        X_h = self.relu(self.bn_h(self.Conv_weight(X_h)))
        return X_h

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


#discriminator
class discriminator(nn.Module):
    def __init__(self, inchannel, outchannel, num_classes, patch_size):
        super(discriminator, self).__init__()
        self.inchannel=inchannel
        print(inchannel)
        self.outchannel = outchannel
        self.num_classes = num_classes
        self.sa = SA(inchannel)
        self.se = SE(inchannel)
        self.patch_size=patch_size
        self.feature_layers = DCRN_02(inchannel, patch_size, patch_size)
        self.head1 = nn.Sequential(nn.Linear(288, 128))
        self.head2 = nn.Sequential(nn.Linear(288, 128))
        self.fc2 = nn.Linear(288, 1)
        self.sigmoid = nn.Sigmoid()

        self.n_outputs = 288
        self.inchannel = inchannel
        self.patch_size = patch_size
        self.Weight_Alpha = nn.Parameter(torch.ones(2) / 2, requires_grad=True)
        self.Spectral_Weight_1 = Spectral_Weight(inchannel, outchannel, kernel_size=3, stride=1, padding=1)
        self.Spatial_Weight_1 = Spatial_Weight(inchannel, outchannel, kernel_size=3, stride=1, padding=1)
        self.mp = nn.MaxPool2d(2)
        self.Spectral_Weight_2 = Spectral_Weight(outchannel, outchannel, kernel_size=3, stride=1, padding=1)
        self.Spatial_Weight_2 = Spatial_Weight(outchannel, outchannel, kernel_size=3, stride=1, padding=1)
        self.Spectral_Weight_3 = Spectral_Weight(outchannel, outchannel, kernel_size=3, stride=1, padding=1)
        self.Spatial_Weight_3 = Spatial_Weight(outchannel, outchannel, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(self._get_final_flattened_size(), outchannel)
        #print("self.fc1.weight shape:", self.fc1.weight.shape)  # self.fc1.weight shape: torch.Size([128, 4608])
        #print("self.fc1.bias shape:", self.fc1.bias.shape)  # self.fc1.bias shape: torch.Size([128])
        self.relu1 = nn.ReLU(inplace=True)
        self.cls_head_src = nn.Linear(outchannel, num_classes)

       # self.feature_layers = DCRN_02(n_band, patch_size, num_class)


    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, self.inchannel,
                             self.patch_size, self.patch_size))
            #print("上边的x.shape", x.shape)
            in_size = x.size(0)
            out1 = self.Spatial_Weight_1(x)
            #print(out1.shape)
            out2 = self.Spatial_Weight_2(out1)
            #print(out2.shape)
            out3 = self.mp(self.Spatial_Weight_3(out2))
            out3 = out3.view(in_size, -1)
            #print(out3.shape)
            w, h = out3.size()
            fc_1 = w * h
            #print("fc_1.shape",fc_1.shape)
        return fc_1

    def forward(self, x):
        #print("x.shape", x.shape)  # x.shape torch.Size([256, 48, 13, 13])
        in_size = x.size(0)  # 256
        #print("in_size的值", in_size)  # 256
        weight_alpha1 = F.softmax(self.Weight_Alpha, dim=0)
        out1 = weight_alpha1[0] * self.Spectral_Weight_1(x) + weight_alpha1[1] * self.Spatial_Weight_1(x)
        #print("out1的形状", out1.shape)  # out1的形状 torch.Size([256, 128, 13, 13])
        weight_alpha2 = F.softmax(self.Weight_Alpha, dim=0)
        out2 = weight_alpha2[0] * self.Spectral_Weight_2(out1) + weight_alpha2[1] * self.Spatial_Weight_2(out1)
        #print("out2的形状", out2.shape)  # out2的形状 torch.Size([256, 128, 13, 13])
        weight_alpha3 = F.softmax(self.Weight_Alpha, dim=0)
        out3 = weight_alpha3[0] * self.Spectral_Weight_3(out2) + weight_alpha3[1] * self.Spatial_Weight_3(out2)
        out3 = self.mp(out3)
        #print("out3的形状", out3.shape)  # out3的形状 torch.Size([256, 128, 6, 6])
        out3 = out3.view(in_size, -1)
        out4 = self.relu1(self.fc1(out3))
        #print("out4.shape",out4.shape)
        clss_01 = self.cls_head_src(out4) #print("clss.shape",clss.shape)#(128,7)
        #print("clss_01.shape", clss_01.shape)#clss_01.shape torch.Size([32, 7])

        features = self.feature_layers(x)
        #print("features.shape", features.shape)#features.shape torch.Size([32, 288])

        x1 = F.normalize(self.head1(features), dim=1)
        #print("x1.shape", x1.shape)#x1.shape torch.Size([32, 128])
        x2 = F.normalize(self.head2(features), dim=1)
        #print("x2.shape", x2.shape)#x2.shape torch.Size([32, 128])

        output = self.fc2(features)  #5*5
        output = self.sigmoid(output)
        #print("output.shape", output.shape)#output.shape torch.Size([32, 1])
        #print("clss01",clss_01.shape)

        return features, x1, x2, clss_01, output


    def get_embedding(self, x):
        features = self.feature_layers(x)
        #print("discriminator_02_shape",features.shape)
        return features

#discriminator_01
class discriminator_01(nn.Module):
    def __init__(self, inchannel, outchannel, num_classes, patch_size):
        super(discriminator_01, self).__init__()
        self.inchannel=inchannel
        print(inchannel)
        self.outchannel = outchannel
        self.num_classes = num_classes
        self.sa = SA(inchannel)
        self.se = SE(inchannel)
        self.patch_size=patch_size
        self.feature_layers = DCRN_02(inchannel, patch_size, patch_size)
        self.head1 = nn.Sequential(nn.Linear(288, 128))
        self.head2 = nn.Sequential(nn.Linear(288, 128))
        self.fc2 = nn.Linear(288, 1)
        self.sigmoid = nn.Sigmoid()

        self.n_outputs = 128
        self.inchannel = inchannel
        self.patch_size = patch_size
        self.Weight_Alpha = nn.Parameter(torch.ones(2) / 2, requires_grad=True)
        self.Spectral_Weight_1 = Spectral_Weight(inchannel, outchannel, kernel_size=3, stride=1, padding=1)
        self.Spatial_Weight_1 = Spatial_Weight(inchannel, outchannel, kernel_size=3, stride=1, padding=1)
        self.mp = nn.MaxPool2d(2)
        self.Spectral_Weight_2 = Spectral_Weight(outchannel, outchannel, kernel_size=3, stride=1, padding=1)
        self.Spatial_Weight_2 = Spatial_Weight(outchannel, outchannel, kernel_size=3, stride=1, padding=1)
        self.Spectral_Weight_3 = Spectral_Weight(outchannel, outchannel, kernel_size=3, stride=1, padding=1)
        self.Spatial_Weight_3 = Spatial_Weight(outchannel, outchannel, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(self._get_final_flattened_size(), outchannel)
        #print("self.fc1.weight shape:", self.fc1.weight.shape)  # self.fc1.weight shape: torch.Size([128, 4608])
        #print("self.fc1.bias shape:", self.fc1.bias.shape)  # self.fc1.bias shape: torch.Size([128])
        self.relu1 = nn.ReLU(inplace=True)
        self.cls_head_src = nn.Linear(outchannel, num_classes)

       # self.feature_layers = DCRN_02(n_band, patch_size, num_class)


    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, self.inchannel,
                             self.patch_size, self.patch_size))
            #print("上边的x.shape", x.shape)
            in_size = x.size(0)
            out1 = self.Spatial_Weight_1(x)
            #print(out1.shape)
            out2 = self.Spatial_Weight_2(out1)
            #print(out2.shape)
            out3 = self.mp(self.Spatial_Weight_3(out2))
            out3 = out3.view(in_size, -1)
            #print(out3.shape)
            w, h = out3.size()
            fc_1 = w * h
            #print("fc_1.shape",fc_1.shape)
        return fc_1

    def forward(self, x):
        #print("x.shape", x.shape)  # x.shape torch.Size([256, 48, 13, 13])
        in_size = x.size(0)  # 256
        #print("in_size的值", in_size)  # 256
        weight_alpha1 = F.softmax(self.Weight_Alpha, dim=0)
        out1 = weight_alpha1[0] * self.Spectral_Weight_1(x) + weight_alpha1[1] * self.Spatial_Weight_1(x)
        #print("out1的形状", out1.shape)  # out1的形状 torch.Size([256, 128, 13, 13])
        weight_alpha2 = F.softmax(self.Weight_Alpha, dim=0)
        out2 = weight_alpha2[0] * self.Spectral_Weight_2(out1) + weight_alpha2[1] * self.Spatial_Weight_2(out1)
        #print("out2的形状", out2.shape)  # out2的形状 torch.Size([256, 128, 13, 13])
        weight_alpha3 = F.softmax(self.Weight_Alpha, dim=0)
        out3 = weight_alpha3[0] * self.Spectral_Weight_3(out2) + weight_alpha3[1] * self.Spatial_Weight_3(out2)
        out3 = self.mp(out3)
        #print("out3的形状", out3.shape)  # out3的形状 torch.Size([256, 128, 6, 6])
        out3 = out3.view(in_size, -1)
        out4 = self.relu1(self.fc1(out3))
        #print("out4.shape",out4.shape)
        clss_01 = self.cls_head_src(out4) #print("clss.shape",clss.shape)#(128,7)
        #print("clss_01.shape", clss_01.shape)#clss_01.shape torch.Size([32, 7])

        features = self.feature_layers(x)
        #print("features.shape", features.shape)#features.shape torch.Size([32, 288])

        x1 = F.normalize(self.head1(features), dim=1)
        #print("x1.shape", x1.shape)#x1.shape torch.Size([32, 128])
        x2 = F.normalize(self.head2(features), dim=1)
        #print("x2.shape", x2.shape)#x2.shape torch.Size([32, 128])

        output = self.fc2(features)
        output = self.sigmoid(output)
        #print("output.shape", output.shape)#output.shape torch.Size([32, 1])
        #print("clss01",clss_01.shape)

        return features, x1, x2, clss_01, output


    def get_embedding(self, x):
        features = self.feature_layers(x)
        #print("discriminator_02_shape",features.shape)
        return features

#DCRN_02
class DCRN_02(nn.Module):

    def __init__(self, input_channels, patch_size, n_classes,channels=288):
        super(DCRN_02, self).__init__()
        #print(input_channels)#48

        self.kernel_dim = 1
        #print("input_channels",input_channels)
        self.feature_dim = input_channels
        #print("patch_size",patch_size)
        self.sz = patch_size
        self.SA = SA(channels)
        self.SE = SE(channels)
        # Convolution Layer 1 kernel_size = (1, 1, 7), stride = (1, 1, 2), output channels = 24
        self.conv1 = nn.Conv3d(1, 24, kernel_size=(7, 1, 1), stride=(2, 1, 1), bias=True)
        self.bn1 = nn.BatchNorm3d(24)
        self.activation1 = nn.ReLU()

        # Residual block 1
        self.conv2 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0),bias=True)  # padding_mode='replicate',
        self.bn2 = nn.BatchNorm3d(24)
        self.activation2 = nn.ReLU()
        self.conv3 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0),
                               bias=True)  # padding_mode='replicate',
        self.bn3 = nn.BatchNorm3d(24)
        self.activation3 = nn.ReLU()


        # Convolution Layer 2 kernel_size = (1, 1, (self.feature_dim - 6) // 2), output channels = 128
        self.conv4 = nn.Conv3d(24, 192, kernel_size=(((self.feature_dim - 7) // 2 + 1), 1, 1), bias=True)
        self.bn4 = nn.BatchNorm3d(192)
        self.activation4 = nn.ReLU()

        #self.conv_dilated = nn.Conv3d(192, 192, kernel_size=(7, 1, 1), stride=1, padding=(6, 0, 0), dilation=2,bias=True)
        #self.bn_dilated = nn.BatchNorm3d(192)
        #self.activation_dilated = nn.ReLU()

        # Finish
        # Convolution layer for spatial information
        self.conv5 = nn.Conv3d(1, 24, (self.feature_dim, 1, 1))
        self.bn5 = nn.BatchNorm3d(24)
        self.activation5 = nn.ReLU()

        # Residual block 2
        self.conv6 = nn.Conv3d(24, 24, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                               bias=True)  # padding_mode='replicate',
        self.bn6 = nn.BatchNorm3d(24)
        self.activation6 = nn.ReLU()
        self.conv7 = nn.Conv3d(24, 96, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                               bias=True)  # padding_mode='replicate',
        self.bn7 = nn.BatchNorm3d(96)
        self.activation7 = nn.ReLU()
        # 膨胀卷积层
        #self.conv_dilated_01 = nn.Conv3d(96, 96, kernel_size=(1, 3, 3), stride=1, padding=(0, 2, 2), dilation=2,bias=True)
        #self.bn_dilated_01 = nn.BatchNorm3d(96)
        #self.activation_dilated_01 = nn.ReLU()

        self.conv8 = nn.Conv3d(24, 96, kernel_size=1)
        # Finish

        # Combination shape
        # self.inter_size = 128 + 24
        self.inter_size = 192 + 96


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

    def forward(self, x):
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
        #print("x1.shape",x1.shape)#x1.shape torch.Size([32, 24, 21, 7, 7])
        x1 = self.conv4(x1)  # (32,128,1,7,7)
        x1 = self.activation4(self.bn4(x1))

        #x1 = self.conv_dilated(x1)
        #x1 = self.activation_dilated(self.bn_dilated(x1))
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

        #x2 = self.conv_dilated_01(x2)
        #x2 = self.activation_dilated_01(self.bn_dilated_01(x2))

        x2 = x2.reshape(x2.size(0), x2.size(1), x2.size(3), x2.size(4))  # (32,24,7,7)

        # concat spatial and spectral information
        x = torch.cat((x1, x2), 1)  # (32,152,7,7)
        #print("x.shape",x.shape)

        ###################
        # attention map
        ###################
        ###################
        # attention map
        ###################
        x = self.ca(x) * x
        #x = self.SA(x, x)
        #x = self.SE(x, x)
        #print("x.self.ca",x.shape)
        x = self.sa(x) * x
        #print("x.self.sa", x.shape)
        #print("x.shape",x.shape)#torch.Size([32, 288, 7, 7])
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)  # (288)

        #####################
        # attention map over
        #####################
        # CMMD

        return x

#DCRN_04
class DCRN_04(nn.Module):
    def __init__(self, input_channels, patch_size, n_classes, channels=288):
        super(DCRN_04, self).__init__()
        self.kernel_dim = 1
        self.feature_dim = input_channels
        self.sz = patch_size
        # Convolution layer 1
        self.conv1 = nn.Conv3d(1, 24, kernel_size=(7, 1, 1), stride=(2, 1, 1), bias=True)
        self.bn1 = nn.BatchNorm3d(24)
        self.activation1 = nn.ReLU()

        # Convolution layer 2
        self.conv_dilated_02 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(6, 0, 0), dilation=2, bias=True)
        self.bn_dilated_02 = nn.BatchNorm3d(24)
        self.activation_dilated_02 = nn.ReLU()

        # Convolution layer 3
        self.conv_dilated_03 = nn.Conv3d(24, 24, kernel_size=(7, 1, 1), stride=1, padding=(15, 0, 0), dilation=5,bias=True)
        self.bn_dilated_03 = nn.BatchNorm3d(24)
        self.activation_dilated_03 = nn.ReLU()
        # Convolution layer 4
        self.conv4 = nn.Conv3d(24, 192, kernel_size=(((self.feature_dim - 7) // 2 + 1), 1, 1), bias=True)
        self.bn4 = nn.BatchNorm3d(192)
        self.activation4 = nn.ReLU()



        self.conv5 = nn.Conv3d(1, 24, (self.feature_dim, 1, 1))
        self.bn5 = nn.BatchNorm3d(24)
        self.activation5 = nn.ReLU()
        # Convolution layer 5
        self.conv_dilated_05 = nn.Conv3d(24, 24, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), dilation=1,
                                         bias=True)
        self.bn_dilated_05 = nn.BatchNorm3d(24)
        self.activation_dilated_05 = nn.ReLU()
        # Convolution layer 6
        self.conv_dilated_06 = nn.Conv3d(24, 96, kernel_size=(1, 3, 3), stride=1, padding=(0, 2, 2), dilation=2,
                                         bias=True)
        self.bn_dilated_06 = nn.BatchNorm3d(96)
        self.activation_dilated_06 = nn.ReLU()
        # Convolution layer 7
        self.conv_dilated_07 = nn.Conv3d(96, 96, kernel_size=(1, 3, 3), stride=1, padding=(0, 5, 5), dilation=5,
                                         bias=True)
        self.bn_dilated_07 = nn.BatchNorm3d(96)
        self.activation_dilated_07 = nn.ReLU()

        self.conv8 = nn.Conv3d(24, 96, kernel_size=1)
        self.inter_size = 192 + 96

        self.conv9 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                               bias=True)
        self.bn9 = nn.BatchNorm3d(self.inter_size)
        self.activation9 = nn.ReLU()
        self.conv10 = nn.Conv3d(self.inter_size, self.inter_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                                bias=True)
        self.bn10 = nn.BatchNorm3d(self.inter_size)
        self.activation10 = nn.ReLU()
        self.ca = ChannelAttention(self.inter_size)
        self.sa = SpatialAttention()

        self.avgpool = nn.AvgPool3d((1, self.sz, self.sz))

        self.fc1 = nn.Linear(in_features=self.inter_size, out_features=n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, C, H, W)#[32, 1, 48, 7, 7])
        #print(x.shape)
        x1 = self.conv1(x)
        x1 = self.activation1(self.bn1(x1))#([32, 24, 21, 7, 7])

        residual = x1
        x1 = self.conv_dilated_02(x1)
        x1 = self.activation_dilated_02(self.bn_dilated_02(x1))#([32, 24, 21, 7, 7])

        x1 = self.conv_dilated_03(x1)
        x1 = residual + x1  #[32, 24, 21, 7, 7])

        x1 = self.activation_dilated_03(self.bn_dilated_03(x1))#[32, 24, 21, 7, 7])

        x1 = self.conv4(x1)
        x1 = self.activation4(self.bn4(x1))#32, 192, 1, 7, 7])

        #print(x1.shape)
        #print(x1.reshape)
        x1 = x1.reshape(x1.size(0), x1.size(1), x1.size(3), x1.size(4))#([32, 192, 7, 7])



        x2 = self.conv5(x)
        x2 = self.activation5(self.bn5(x2))  # ([32, 192, 7, 7])
        residual = x2
        residual = self.conv8(residual)
        x2 = self.conv_dilated_05(x2)
        x2 = self.activation_dilated_05(self.bn_dilated_05(x2))
        x2 = self.conv_dilated_06(x2)
        x2 = residual + x2
        x2 = self.activation_dilated_06(self.bn_dilated_06(x2))
        x2 = self.conv_dilated_07(x2)
        x2 = self.activation_dilated_07(self.bn_dilated_07(x2))
        x2 = x2.reshape(x2.size(0), x2.size(1), x2.size(3), x2.size(4))#[32, 96, 7, 7])
        #print(x2.shape)
        x = torch.cat((x1, x2), 1)  # (32,288,7,7)
        #print(x.shape)
        # print(x.shape)
        ###################
        # attention map
        ###################
        ###################
        # attention map
        ###################
        x = self.ca(x) * x
        x = self.sa(x) * x
        #print(x.shape)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)  # (288)

        #####################
        # attention map over
        #####################
        # CMMD
        #print(x.shape)
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