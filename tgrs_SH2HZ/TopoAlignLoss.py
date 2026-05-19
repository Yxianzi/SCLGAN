import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
warnings.filterwarnings('ignore')


class ImprovedTopoAlignLoss(nn.Module):
    def __init__(self, base_k=2, max_k=16, temp_base=0.4, feat_dim=128,memory_size=32):
        super().__init__()
        self.feat_dim = feat_dim
        self.base_k = base_k
        self.max_k = max_k
        self.temp_base = temp_base
        self.memory_size = memory_size
        self.register_buffer("tgt_memory", torch.zeros(memory_size, feat_dim))
        self._init_memory(seed=42)  # 固定初始化种子

        # 控制变量
        self.register_buffer("memory_ptr", torch.tensor(0))
        self.register_buffer("current_step", torch.tensor(0))

    def _init_memory(self, seed):
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            self.tgt_memory.normal_(mean=0, std=0.1)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)

    def update_epoch(self, epoch):
        self.current_epoch = epoch

    def adaptive_k(self):
        return min(self.base_k + self.current_epoch // 5, self.max_k)

    def adaptive_temperature(self, k):
        return max(0.1, self.temp_base * (1 + math.log(k + 1)))

    def update_memory(self, src, tgt):
        with torch.no_grad():
            batch_size = src.size(0)
            ptr = self.memory_ptr.item()
            momentum = min(0.5, 0.3 + self.current_step * 0.01)
            for i in range(batch_size):
                idx = (ptr + i) % self.memory_size
                self.tgt_memory[idx] = (
                        momentum * F.normalize(tgt[i], dim=0) +
                        (1 - momentum) * self.tgt_memory[idx]
                )
            self.memory_ptr.fill_((ptr + batch_size) % self.memory_size)
            self.current_step += 1
    # 2
    def forward(self, src_feat, tgt_feat):
        assert src_feat.size(0) == tgt_feat.size(0), ""
        batch_size = src_feat.size(0)

        src_norm = F.normalize(src_feat, dim=1, p=2, eps=1e-6)
        tgt_norm = F.normalize(tgt_feat, dim=1, p=2, eps=1e-6)
        mem_tgt_norm = F.normalize(self.tgt_memory, dim=1, p=2, eps=1e-6)

        sim_current = torch.mm(src_norm, tgt_norm.t())  # [B,B]

        sim_src_memtgt = torch.mm(src_norm, mem_tgt_norm.t())  # [B,M]

        sim_matrix = torch.cat([sim_current, sim_src_memtgt], dim=1)  # [B, B+M]

        k = self.adaptive_k()
        temperature = self.adaptive_temperature(k)

        scaled_sim = sim_matrix / temperature
        topk_values, topk_indices = torch.topk(scaled_sim, k=k, dim=1)

        is_memory = topk_indices >= batch_size

        memory_indices = (topk_indices - batch_size).clamp(min=0, max=sim_src_memtgt.shape[1] - 1)
        batch_indices = topk_indices.clamp(max=sim_current.shape[1] - 1)

        match_scores = torch.where(
            is_memory,
            torch.gather(sim_src_memtgt, 1, memory_indices),
            torch.gather(sim_current, 1, batch_indices)
        )
        weights = torch.softmax(topk_values / temperature, dim=1)
        loss = (weights * (1 - match_scores)).sum(dim=1).mean()

        self.update_memory(src_feat.detach(), tgt_feat.detach())
        return loss












