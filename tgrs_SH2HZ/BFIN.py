import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleConsistency(nn.Module):
    def __init__(self, scales=[1, 3, 5]):
        super().__init__()
        self.scales = scales
        self.pools = nn.ModuleList([
            nn.AvgPool1d(kernel_size=s, stride=1, padding=s // 2)
            for s in scales
        ])

    def forward(self, features):

        if features.dim() == 2:
            features = features.unsqueeze(-1)  # [B, D, 1]

        consistency = []
        for pool in self.pools:
            smoothed = pool(features)  # [B, D, 1]
            diff = (smoothed - features).abs().mean(dim=1)  # [B, 1]
            consistency.append(1 - diff.squeeze())

        return torch.stack(consistency, dim=1).mean(dim=1)  # [B]

class SinkhornDistance(nn.Module):
    def __init__(self, eps=0.1, max_iter=3, stable=True, scale_factor=0.01):

        super().__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.stable = stable
        self.scale_factor = scale_factor

    def forward(self, x, y):
        x_norm = F.normalize(x, p=2, dim=1)  # L2归一化
        y_norm = F.normalize(y, p=2, dim=1)  # L2归一化

        cost_matrix = torch.cdist(x_norm, y_norm, p=2)  # (B, C)
        cost_matrix = cost_matrix * self.scale_factor

        if self.stable:
            log_K = -cost_matrix / (self.eps + 1e-8)
            B, C = cost_matrix.shape
            log_u = torch.zeros(B, device=x.device)
            log_v = torch.zeros(C, device=y.device)

            for _ in range(self.max_iter):
                log_v = -torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
                log_u = -torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)

            transport_plan = torch.exp(log_K + log_u.unsqueeze(1) + log_v.unsqueeze(0))
        else:
            K = torch.exp(-cost_matrix / (self.eps + 1e-8))
            B, C = cost_matrix.shape
            u = torch.ones(B, device=x.device) / B
            for _ in range(self.max_iter):
                v = 1.0 / (K.T @ u + 1e-8)
                u = 1.0 / (K @ v + 1e-8)
            transport_plan = u.unsqueeze(1) * K * v.unsqueeze(0)
        distance = (transport_plan * cost_matrix).sum()
        return distance, transport_plan

class bfin(nn.Module):
    def __init__(self, cnn_dim, former_dim, num_dim, batch,n_classes,k_init=10.0,
                 base_retention=(1.0, 1.0), retention_weight=0.01):
        super(bfin, self).__init__()
        self.cnn_dim = cnn_dim
        self.former_dim = former_dim
        self.num_dim = num_dim
        self.retention_weight = retention_weight
        self.dim = self.cnn_dim
        self.num_heads = self.dim // batch
        self.class_prototypes_cnn = nn.Parameter(torch.randn(num_dim, cnn_dim))
        self.class_prototypes_former = nn.Parameter(torch.randn(num_dim, former_dim))
        self.sinkhorn = SinkhornDistance()
        self.ms_consistency = MultiScaleConsistency()
        self.gate_cnn = nn.Sequential(
            nn.Linear(cnn_dim + former_dim, cnn_dim),
            nn.ReLU(),
            nn.Linear(cnn_dim, 1),
            nn.Sigmoid()
        )
        self.gate_former = nn.Sequential(
            nn.Linear(cnn_dim + former_dim, former_dim),
            nn.ReLU(),
            nn.Linear(former_dim, 1),
            nn.Sigmoid()
        )
        self.threshold_initialized = False
        self.threshold_cnn = nn.Parameter(torch.tensor(float(self.threshold_initialized)))
        self.threshold_former = nn.Parameter(torch.tensor(float(self.threshold_initialized)))
        self.log_k_cnn = nn.Parameter(torch.log(torch.tensor(float(k_init))))
        self.log_k_former = nn.Parameter(torch.log(torch.tensor(float(k_init))))
        self.register_buffer('base_retention_cnn', torch.tensor(base_retention[0]))
        self.register_buffer('base_retention_former', torch.tensor(base_retention[1]))

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, num_dim)
        )
        self.attention_gate = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.GELU()
        )
        self.ssca = SSCA(feat_dim=cnn_dim + former_dim, num_heads=self.num_heads,spectral_bins=32)

    def _initialize_threshold(self, first_batch_features):
        with torch.no_grad():
            mean = first_batch_features.abs().mean()
            std = first_batch_features.abs().std()
            init_value = torch.sigmoid((mean - 0.5 * std) * 3 - 1).item()
            self.threshold_cnn.data.fill_(init_value)
            self.threshold_former.data.fill_(init_value)

    def wasserstein_importance(self, features, prototypes):
        distance, _ = self.sinkhorn(features, prototypes)
        importance = torch.exp(-distance)
        importance = torch.clamp(importance, min=1e-8, max=1.0)
        return importance

    def get_dynamic_retention(self, cnn_feat, former_feat):
        cnn_imp = self.wasserstein_importance(cnn_feat, self.class_prototypes_cnn)
        former_imp = self.wasserstein_importance(former_feat, self.class_prototypes_former)
        cnn_consistency = self.ms_consistency(cnn_feat)
        former_consistency = self.ms_consistency(former_feat)
        cnn_imp = cnn_imp * 0.6 + cnn_consistency * 0.4
        former_imp = former_imp * 0.6 + former_consistency * 0.4
        imp_ratio = (former_imp.mean() / (cnn_imp.mean() + 1e-6))
        cnn_features = self.base_retention_cnn * imp_ratio
        former_features = self.base_retention_former * imp_ratio
        return cnn_features.clamp(0.05, 0.95), former_features.clamp(0.05, 0.95), imp_ratio.item()


    def forward(self, cnn_features_s, former_features_s,cnn_features_t, former_features_t,false = 'train'):

        if false == 'test':
            combined_t = torch.cat([cnn_features_t, former_features_t], dim=1)
            if not self.threshold_initialized:
                self._initialize_threshold(combined_t)
            target_cnn_t, target_former_t, _ = self.get_dynamic_retention(cnn_features_t, former_features_t)

            k_cnn_t = torch.exp(self.log_k_cnn).clamp(1.0, 20.0)
            k_former_t = torch.exp(self.log_k_former).clamp(1.0, 20.0)

            thres_cnn_t = torch.sigmoid(self.threshold_cnn)
            thres_former_t = torch.sigmoid(self.threshold_former)

            gate_cnn_t = self.gate_cnn(combined_t)
            gate_former_t = self.gate_former(combined_t)

            cnn_mask_t = torch.sigmoid(
                k_cnn_t * ( gate_cnn_t-thres_cnn_t) +
                (target_cnn_t - 0.5) * 2.0
            ).clamp(0, 1)

            former_mask_t = torch.sigmoid(
                k_former_t * (gate_former_t-thres_former_t) +
                (target_former_t - 0.5) * 2.0
            ).clamp(0, 1)

            cnn_filtered_t = cnn_features_t * cnn_mask_t
            former_filtered_t = former_features_t * former_mask_t
            retention_loss_t = F.mse_loss(cnn_mask_t.mean(), target_cnn_t) + \
                               F.mse_loss(former_mask_t.mean(), target_former_t)

            y = torch.cat((cnn_filtered_t.unsqueeze(1), former_filtered_t.unsqueeze(1)), dim=1)  # (B, 2, D)
            y = self.attention_gate(y)  # (B, 2, D)
            token_t = y[:, 0]  # (B, D)
            cls_output_t = self.mlp_head(token_t)

            return cls_output_t, cnn_mask_t, former_mask_t, self.retention_weight * retention_loss_t

        else:
            combined_ss = torch.cat([cnn_features_s, former_features_s], dim=1)
            combined_tt = torch.cat([cnn_features_t, former_features_t], dim=1)
            combined_s,sparsity_loss_s = self.ssca(combined_ss, combined_tt)
            combined_t,sparsity_loss_t = self.ssca(combined_tt, combined_ss)

            if not self.threshold_initialized:
                self._initialize_threshold(combined_s)
            gate_cnn_s = self.gate_cnn(combined_s)
            gate_former_s = self.gate_former(combined_s)
            k_cnn_s = torch.exp(self.log_k_cnn).clamp(1.0, 20.0)
            k_former_s = torch.exp(self.log_k_former).clamp(1.0, 20.0)

            thres_cnn_s = torch.sigmoid(self.threshold_cnn)
            thres_former_s = torch.sigmoid(self.threshold_former)

            source_cnn_s, source_former_s, imp_ratio_s = self.get_dynamic_retention(cnn_features_s, former_features_s)
            cnn_mask_s = torch.sigmoid(
                k_cnn_s * ( gate_cnn_s-thres_cnn_s) +
                (source_cnn_s - 0.5) * 2.0
            ).clamp(0, 1)

            former_mask_s = torch.sigmoid(
                k_former_s * (gate_former_s- thres_former_s) +
                (source_former_s - 0.5) * 2.0
            ).clamp(0, 1)

            if not self.threshold_initialized:
                self._initialize_threshold(combined_t)
            gate_cnn_t = self.gate_cnn(combined_t)
            gate_former_t = self.gate_former(combined_t)
            target_cnn_t, target_former_t, imp_ratio_t = self.get_dynamic_retention(cnn_features_t, former_features_t)

            k_cnn_t = torch.exp(self.log_k_cnn).clamp(1.0, 20.0)
            k_former_t = torch.exp(self.log_k_former).clamp(1.0, 20.0)

            thres_cnn_t = torch.sigmoid(self.threshold_cnn)
            thres_former_t = torch.sigmoid(self.threshold_former)
            cnn_mask_t = torch.sigmoid(
                k_cnn_t * ( gate_cnn_t-thres_cnn_t) +
                (target_cnn_t - 0.5) * 2.0
            ).clamp(0, 1)

            former_mask_t = torch.sigmoid(
                k_former_t * (gate_former_t- thres_former_t) +
                (target_former_t - 0.5) * 2.0
            ).clamp(0, 1)

            cnn_filtered_s = cnn_features_s * cnn_mask_s
            former_filtered_s = former_features_s * former_mask_s
            cnn_filtered_t = cnn_features_t * cnn_mask_t
            former_filtered_t = former_features_t * former_mask_t

            retention_loss_s = F.mse_loss(cnn_mask_s.mean(), source_cnn_s) + F.mse_loss(former_mask_s.mean(), source_former_s)
            retention_loss_t = F.mse_loss(cnn_mask_t.mean(), target_cnn_t) + F.mse_loss(former_mask_t.mean(), target_former_t)

            x = torch.cat((cnn_filtered_s.unsqueeze(1), former_filtered_s.unsqueeze(1)), dim=1)  # (B, 2, D)

            x = self.attention_gate(x)  # (B, 2, D)
            token = x[:, 0]  # (B, D)
            cls_output_s = self.mlp_head(token)

            y = torch.cat((cnn_filtered_t.unsqueeze(1), former_filtered_t.unsqueeze(1)), dim=1)  # (B, 2, D)
            y = self.attention_gate(y)  # (B, 2, D)
            token_t = y[:, 0]  # (B, D)
            cls_output_t = self.mlp_head(token_t)
            return cls_output_s, cnn_mask_s, former_mask_s,  self.retention_weight * retention_loss_s + 0.5 * sparsity_loss_s,\
                   cls_output_t, cnn_mask_t, former_mask_t, self.retention_weight * retention_loss_t+ 0.5 * sparsity_loss_t


class SSCA(nn.Module):
    def __init__(self, spectral_bins, feat_dim, num_heads, expansion_ratio=4, sparsity_lambda=0.1):
        super().__init__()

        self.spatial_dim = feat_dim // spectral_bins
        self.spectral_bins = spectral_bins
        self.feat_dim = feat_dim
        self.sparsity_lambda = sparsity_lambda  # 稀疏化系数

        self.spectral_attn = nn.MultiheadAttention(
            embed_dim=self.spatial_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # Spatial attention
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(
                spectral_bins, spectral_bins,
                kernel_size=3, padding=1,
                groups=spectral_bins
            ),
            nn.BatchNorm2d(spectral_bins),
            nn.GELU()
        )

        self.cross_fusion = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim * expansion_ratio),
            nn.GELU(),
            nn.Linear(feat_dim * expansion_ratio, feat_dim),
            nn.Dropout(0.1)
        )

        self.spectral_gate = nn.Sequential(
            nn.Linear(spectral_bins, spectral_bins),
            nn.Sigmoid()
        )

        self.reassemble = nn.Linear(feat_dim + spectral_bins, feat_dim)
        self.norm = nn.LayerNorm(feat_dim)


    def forward(self, src_feat, tgt_feat):

        B_src, D_src = src_feat.shape
        B_tgt, D_tgt = tgt_feat.shape

        original_src_feat = src_feat
        if B_src != B_tgt:
            min_B = min(B_src, B_tgt)
            src_feat = src_feat[:min_B]
            tgt_feat = tgt_feat[:min_B]
            B_src = B_tgt = min_B

        src_spectral = src_feat.view(B_src, self.spectral_bins, self.spatial_dim)
        tgt_spectral = tgt_feat.view(B_tgt, self.spectral_bins, self.spatial_dim)

        attn_output, attn_weights = self.spectral_attn(
            query=src_spectral,
            key=tgt_spectral,
            value=tgt_spectral
        )
        spectral_out = attn_output * self.temperature

        sparsity_loss = self.sparsity_lambda * attn_weights.abs().mean()

        spatial_out = self.spatial_attn(
            src_spectral.unsqueeze(-1)
        ).squeeze(-1)

        gate = self.spectral_gate(
            spectral_out.mean(dim=-1)
        ).unsqueeze(-1)
        fused_spectral = gate * spectral_out + (1 - gate) * spatial_out

        cross_feat = self.cross_fusion(
            torch.cat([src_feat, fused_spectral.reshape(B_src, -1)], dim=1)
        )

        output = self.reassemble(
            torch.cat([cross_feat, gate.squeeze(-1)], dim=1)
        )

        if B_src != original_src_feat.shape[0]:
            output = self.norm(output + src_feat)
            remaining = original_src_feat[B_src:]
            output = torch.cat([output, remaining], dim=0)
        else:
            output = self.norm(output + original_src_feat)

        return output, sparsity_loss

