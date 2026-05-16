import copy
import math

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from ssg_tdus import (
    apply_generation_reliability_gate,
    compute_generation_reliability,
    fuse_tdus_score_with_generation,
)


_LAST_CBP_TDUS_CONFIG = {}


def _remember_cbp_tdus_config(config):
    global _LAST_CBP_TDUS_CONFIG
    _LAST_CBP_TDUS_CONFIG = dict(config)
    return config


def get_cbp_tdus_config(dataset_name):
    name = dataset_name.lower()

    if name == "houston":
        return _remember_cbp_tdus_config({
            "query_ratio": 0.01,
            "max_query": 256,
            "min_conf_start": 0.90,
            "min_conf_floor": 0.80,
            "min_conf_decay": 0.003,
            "min_agree": 0.5,
            "quality_quantile": 0.50,
            "min_selected": 32,
            "min_coverage_ratio": 0.40,
            "max_prior_thr": 0.75,
            "prior_alpha": 0.3,
        })

    if name == "up2pc":
        return _remember_cbp_tdus_config({
            "query_ratio": 0.006,
            "max_query": 256,
            "min_conf_start": 0.96,
            "min_conf_floor": 0.86,
            "min_conf_decay": 0.003,
            "min_agree": 0.5,
            "quality_quantile": 0.70,
            "min_selected": 32,
            "min_coverage_ratio": 0.50,
            "max_prior_thr": 0.60,
            "min_class_count_abs": 2,
            "prior_alpha": 0.2,
        })

    if name == "sh2hz":
        return _remember_cbp_tdus_config({
            "query_ratio": 0.008,
            "max_query": 384,
            "min_conf_start": 0.95,
            "min_conf_floor": 0.82,
            "min_conf_decay": 0.004,
            "min_agree": 1.0,
            "quality_quantile": 0.70,
            "min_selected": 48,
            "min_coverage_ratio": 0.50,
            "max_prior_thr": 0.65,
            "prior_alpha": 0.5,
        })

    return _remember_cbp_tdus_config({
        "query_ratio": 0.006,
        "max_query": 256,
        "min_conf_start": 0.96,
        "min_conf_floor": 0.86,
        "min_conf_decay": 0.003,
        "min_agree": 1.0,
        "quality_quantile": 0.75,
        "min_selected": 32,
        "min_coverage_ratio": 0.50,
        "max_prior_thr": 0.60,
        "prior_alpha": 0.5,
    })


class ModelEMA:
    def __init__(self, model, decay=0.99):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        for param in self.ema.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        model_state = model.state_dict()
        for name, ema_value in self.ema.state_dict().items():
            model_value = model_state[name].detach()
            if torch.is_floating_point(ema_value):
                ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)
        self.ema.eval()


@torch.no_grad()
def compute_source_prototypes(model, source_loader, num_classes, device):
    was_training = model.training
    model.eval()

    sums = None
    counts = torch.zeros(num_classes, device=device)
    global_sum = None
    global_count = 0

    for data, labels in source_loader:
        data = data.to(device).float()
        labels = labels.to(device).long()

        features, _, _, logits, _ = model(data)
        features = features.reshape(features.size(0), -1)

        if sums is None:
            feature_dim = features.size(1)
            sums = torch.zeros(num_classes, feature_dim, device=device)
            global_sum = torch.zeros(feature_dim, device=device)

        global_sum += features.sum(dim=0)
        global_count += features.size(0)

        for class_id in range(num_classes):
            class_mask = labels == class_id
            if class_mask.any():
                sums[class_id] += features[class_mask].sum(dim=0)
                counts[class_id] += class_mask.sum()

    if sums is None:
        prototypes = torch.zeros(num_classes, 1, device=device)
    else:
        prototypes = torch.zeros_like(sums)
        if global_count > 0:
            fallback = global_sum / float(global_count)
        else:
            fallback = torch.zeros(sums.size(1), device=device)

        for class_id in range(num_classes):
            if counts[class_id] > 0:
                prototypes[class_id] = sums[class_id] / counts[class_id].clamp_min(1.0)
            else:
                prototypes[class_id] = fallback

    if was_training:
        model.train()

    return F.normalize(prototypes, dim=1)


def strong_augment_torch(x, noise_std=0.01):
    out = x.clone()
    if torch.rand((), device=out.device) < 0.5:
        out = torch.flip(out, dims=[-1])
    if torch.rand((), device=out.device) < 0.5:
        out = torch.flip(out, dims=[-2])
    if noise_std > 0:
        out = out + torch.randn_like(out) * noise_std
    return out


def _empty_tdus_dataset(data_all=None, pseudo_all=None, score_all=None):
    if data_all is None:
        empty_data = torch.empty(0)
        empty_label = torch.empty(0, dtype=torch.long)
        empty_weight = torch.empty(0, dtype=torch.float)
    else:
        empty_data = data_all[:0].float()
        empty_label = pseudo_all[:0].long()
        empty_weight = score_all[:0].float()
    return TensorDataset(empty_data, empty_label, empty_weight)


def _candidate_score_threshold(score_all, candidate_top_ratio):
    candidate_top_ratio = float(max(0.0, min(1.0, candidate_top_ratio)))
    if score_all.numel() == 0 or candidate_top_ratio <= 0.0:
        return float("inf")
    top_count = max(1, int(math.ceil(score_all.numel() * candidate_top_ratio)))
    top_values = torch.topk(score_all, k=min(top_count, score_all.numel()), largest=True).values
    return float(top_values.min().item())


def _build_candidate_dataset_and_info(
        data_all,
        pseudo_all,
        conf_all,
        entropy_all,
        score_all,
        core_mask,
        num_classes,
        candidate_min_conf,
        candidate_max_entropy,
        candidate_top_ratio,
        extra_candidate_mask=None):
    candidate_threshold = _candidate_score_threshold(score_all, candidate_top_ratio)
    candidate_mask = (
        (conf_all >= float(candidate_min_conf))
        & (entropy_all <= float(candidate_max_entropy))
        & (score_all >= candidate_threshold)
        & (~core_mask.bool())
    )
    if extra_candidate_mask is not None:
        candidate_mask = candidate_mask | (extra_candidate_mask.bool() & (~core_mask.bool()))
    candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).view(-1)
    candidate_pseudo = pseudo_all[candidate_indices].long()
    candidate_hist = torch.bincount(candidate_pseudo, minlength=num_classes).tolist()
    candidate_dataset = TensorDataset(
        data_all[candidate_indices].float(),
        candidate_pseudo,
        score_all[candidate_indices].float()
    )
    return candidate_dataset, {
        "num_candidate": int(candidate_indices.numel()),
        "candidate_hist": [int(x) for x in candidate_hist],
        "candidate_coverage": int(sum(1 for x in candidate_hist if int(x) > 0)),
        "candidate_min_conf": float(candidate_min_conf),
        "candidate_max_entropy": float(candidate_max_entropy),
        "candidate_top_ratio": float(candidate_top_ratio),
        "candidate_score_threshold": candidate_threshold,
    }


def _empty_candidate_info(num_classes, candidate_min_conf, candidate_max_entropy, candidate_top_ratio):
    return {
        "num_candidate": 0,
        "candidate_hist": [0 for _ in range(num_classes)],
        "candidate_coverage": 0,
        "candidate_min_conf": float(candidate_min_conf),
        "candidate_max_entropy": float(candidate_max_entropy),
        "candidate_top_ratio": float(candidate_top_ratio),
        "candidate_score_threshold": 0.0,
    }


def _attach_candidate_info(info, candidate_dataset, candidate_info, total_count, core_count, num_classes):
    info.update(candidate_info)
    info["candidate_dataset"] = candidate_dataset
    info["num_core"] = int(core_count)
    core_hist = [int(x) for x in info.get("core_hist", [0 for _ in range(num_classes)])]
    pre_core_hist = [int(x) for x in info.get("pre_core_hist", core_hist)]
    info["core_hist"] = core_hist
    info["pre_core_hist"] = pre_core_hist
    info["class_hist"] = core_hist
    info["coverage"] = int(sum(1 for x in core_hist if int(x) > 0))
    info["pre_core_coverage"] = int(sum(1 for x in pre_core_hist if int(x) > 0))
    info["missing_core_classes"] = [class_id for class_id, count in enumerate(core_hist) if int(count) == 0]
    info["missing_candidate_classes"] = [
        class_id for class_id, count in enumerate(candidate_info["candidate_hist"]) if int(count) == 0
    ]
    info["num_unselected"] = int(max(0, total_count - int(core_count) - candidate_info["num_candidate"]))
    return info


def _attach_generation_info(
        info,
        active_gen_reliability=False,
        gen_rel_all=None,
        gen_agree_all=None,
        gen_prob_cons_all=None,
        gen_feat_cons_all=None,
        gen_quality_all=None,
        base_score_all=None,
        final_score_all=None,
        gen_rel_weight=0.0,
        gen_rel_mode="gate_only",
        gen_gate_strict=False,
        gen_min_agreement=0.67,
        gen_min_prob_consistency=0.50,
        gen_min_quality=0.30,
        core_before_gen_gate=0,
        core_after_gen_gate=0,
        downgraded_to_candidate=0,
        core_low_agree=0,
        core_low_prob_cons=0,
        core_low_quality=0):
    def _mean_or_zero(x):
        return float(x.mean().item()) if x is not None and x.numel() > 0 else 0.0

    def _min_or_zero(x):
        return float(x.min().item()) if x is not None and x.numel() > 0 else 0.0

    def _max_or_zero(x):
        return float(x.max().item()) if x is not None and x.numel() > 0 else 0.0

    gen_agree_all = gen_agree_all if gen_agree_all is not None else torch.empty(0)
    gen_prob_cons_all = gen_prob_cons_all if gen_prob_cons_all is not None else torch.empty(0)
    gen_quality_all = gen_quality_all if gen_quality_all is not None else torch.empty(0)
    info.update({
        "use_gen_reliability": bool(active_gen_reliability),
        "gen_rel_weight": float(gen_rel_weight) if active_gen_reliability else 0.0,
        "gen_rel_mode": gen_rel_mode if active_gen_reliability else "disabled",
        "gen_gate_strict": bool(gen_gate_strict),
        "gen_rel_mean": _mean_or_zero(gen_rel_all),
        "gen_rel_min": _min_or_zero(gen_rel_all),
        "gen_rel_max": _max_or_zero(gen_rel_all),
        "gen_agree_mean": _mean_or_zero(gen_agree_all),
        "gen_prob_cons_mean": _mean_or_zero(gen_prob_cons_all),
        "gen_feat_cons_mean": _mean_or_zero(gen_feat_cons_all),
        "gen_quality_mean": _mean_or_zero(gen_quality_all),
        "low_agree": int((gen_agree_all < float(gen_min_agreement)).sum().item()) if gen_agree_all.numel() > 0 else 0,
        "low_prob_cons": int((gen_prob_cons_all < float(gen_min_prob_consistency)).sum().item()) if gen_prob_cons_all.numel() > 0 else 0,
        "low_quality": int((gen_quality_all < float(gen_min_quality)).sum().item()) if gen_quality_all.numel() > 0 else 0,
        "base_score_mean": _mean_or_zero(base_score_all),
        "final_score_mean": _mean_or_zero(final_score_all),
        "core_before_gen_gate": int(core_before_gen_gate),
        "core_after_gen_gate": int(core_after_gen_gate),
        "downgraded_to_candidate": int(downgraded_to_candidate),
        "core_low_agree": int(core_low_agree),
        "core_low_prob_cons": int(core_low_prob_cons),
        "core_low_quality": int(core_low_quality),
    })
    return info


@torch.no_grad()
def build_cbp_tdus_dataset(
    student_model,
    teacher_model,
    target_loader,
    source_prototypes,
    num_classes,
    batch_size,
    device,
    query_ratio=0.01,
    max_query=512,
    min_conf=0.75,
    min_agree=0.5,
    quality_quantile=0.75,
    min_selected=32,
    min_coverage_ratio=0.50,
    max_prior_thr=0.60,
    min_class_count_abs=None,
    prior_alpha=0.5,
    use_spatial=True,
    spatial_window=3,
    min_spatial_agree=0.50,
    spatial_weight=0.20,
    target_rows=None,
    target_cols=None,
    target_height=None,
    target_width=None,
    use_multiview=True,
    candidate_min_conf=0.80,
    candidate_max_entropy=0.80,
    candidate_top_ratio=0.20,
    generator_model=None,
    use_gen_reliability=True,
    gen_rel_weight=0.30,
    gen_rel_rec_weight=1.0,
    gen_rel_sam_weight=1.0,
    gen_rel_w_agree=0.35,
    gen_rel_w_prob=0.30,
    gen_rel_w_feat=0.20,
    gen_rel_w_quality=0.15,
    gen_rel_mode="gate_only",
    gen_gate_strict=False,
    gen_min_agreement=0.67,
    gen_min_prob_consistency=0.50,
    gen_min_quality=0.30,
):
    student_was_training = student_model.training
    student_model.eval()
    teacher_model.eval()
    generator_was_training = False
    if generator_model is not None:
        generator_was_training = generator_model.training
        generator_model.eval()

    prototypes_norm = F.normalize(source_prototypes.to(device).float(), dim=1)

    all_data = []
    all_pseudo = []
    all_conf = []
    all_entropy = []
    all_proto_dist = []
    all_agree = []
    all_score = []
    all_topology_gap = []
    all_gen_rel = []
    all_gen_agree = []
    all_gen_prob_cons = []
    all_gen_feat_cons = []
    all_gen_quality = []

    for batch in target_loader:
        data = batch[0] if isinstance(batch, (tuple, list)) else batch
        data = data.to(device).float()

        features, _, _, logits, _ = teacher_model(data)
        features = features.reshape(features.size(0), -1)
        prob = F.softmax(logits, dim=1)
        conf, pseudo = prob.max(dim=1)
        entropy = -torch.sum(prob * torch.log(prob + 1e-8), dim=1) / math.log(num_classes)

        features_norm = F.normalize(features, dim=1)
        proto_sim = features_norm @ prototypes_norm.t()
        proto_prob = F.softmax(proto_sim / 0.1, dim=1)
        proto_dist = 1.0 - proto_sim[torch.arange(data.size(0), device=device), pseudo]
        topology_gap = torch.sum(torch.abs(prob - proto_prob), dim=1) / num_classes

        if use_multiview:
            pred1 = pseudo
            view2 = strong_augment_torch(data)
            view3 = strong_augment_torch(data)
            _, _, _, logits2, _ = teacher_model(view2)
            _, _, _, logits3, _ = teacher_model(view3)
            pred2 = torch.argmax(F.softmax(logits2, dim=1), dim=1)
            pred3 = torch.argmax(F.softmax(logits3, dim=1), dim=1)
            agreement = ((pred1 == pred2).float() + (pred1 == pred3).float()) / 2.0
        else:
            agreement = torch.ones_like(conf)

        base_score = (
            conf
            - 0.50 * entropy
            - 0.35 * proto_dist
            - 0.20 * topology_gap
            + 0.20 * agreement
        )

        if use_gen_reliability and generator_model is not None:
            img_spe, img_spa = generator_model(data)
            img_mix = (img_spe + img_spa + data) / 3.0
            feat_spe, _, _, logits_spe, _ = teacher_model(img_spe)
            feat_spa, _, _, logits_spa, _ = teacher_model(img_spa)
            feat_mix, _, _, logits_mix, _ = teacher_model(img_mix)
            prob_spe = F.softmax(logits_spe, dim=1)
            prob_spa = F.softmax(logits_spa, dim=1)
            prob_mix = F.softmax(logits_mix, dim=1)
            gen_info = compute_generation_reliability(
                prob_ori=prob,
                prob_spe=prob_spe,
                prob_spa=prob_spa,
                prob_mix=prob_mix,
                feat_ori=features,
                feat_spe=feat_spe,
                feat_spa=feat_spa,
                feat_mix=feat_mix,
                img_ori=data,
                img_spe=img_spe,
                img_spa=img_spa,
                img_mix=img_mix,
                rec_weight=gen_rel_rec_weight,
                sam_weight=gen_rel_sam_weight,
                w_agree=gen_rel_w_agree,
                w_prob=gen_rel_w_prob,
                w_feat=gen_rel_w_feat,
                w_quality=gen_rel_w_quality,
            )
            gen_reliability = gen_info["gen_reliability"]
        else:
            gen_reliability = torch.ones_like(conf)
            gen_info = {
                "agreement": torch.ones_like(conf),
                "prob_consistency": torch.ones_like(conf),
                "feat_consistency": torch.ones_like(conf),
                "gen_quality": torch.ones_like(conf),
            }

        all_data.append(data.detach().cpu())
        all_pseudo.append(pseudo.detach().cpu())
        all_conf.append(conf.detach().cpu())
        all_entropy.append(entropy.detach().cpu())
        all_proto_dist.append(proto_dist.detach().cpu())
        all_agree.append(agreement.detach().cpu())
        all_score.append(base_score.detach().cpu())
        all_topology_gap.append(topology_gap.detach().cpu())
        all_gen_rel.append(gen_reliability.detach().cpu())
        all_gen_agree.append(gen_info["agreement"].detach().cpu())
        all_gen_prob_cons.append(gen_info["prob_consistency"].detach().cpu())
        gen_feat_cons = gen_info["feat_consistency"]
        gen_quality = gen_info["gen_quality"]
        if gen_feat_cons is None:
            gen_feat_cons = torch.ones_like(conf)
        if gen_quality is None:
            gen_quality = torch.ones_like(conf)
        all_gen_feat_cons.append(gen_feat_cons.detach().cpu())
        all_gen_quality.append(gen_quality.detach().cpu())

    if student_was_training:
        student_model.train()
    if generator_model is not None and generator_was_training:
        generator_model.train()
    teacher_model.eval()

    if not all_data:
        empty_data = torch.empty(0)
        empty_label = torch.empty(0, dtype=torch.long)
        empty_weight = torch.empty(0, dtype=torch.float)
        empty_dataset = TensorDataset(empty_data, empty_label, empty_weight)
        prior_alpha = float(max(0.0, min(1.0, prior_alpha)))
        mixed_prior = [1.0 / num_classes for _ in range(num_classes)]
        quota_per_class = [0 for _ in range(num_classes)]
        info = {
            "num_selected": 0,
            "class_hist": [0 for _ in range(num_classes)],
            "core_hist": [0 for _ in range(num_classes)],
            "pre_core_hist": [0 for _ in range(num_classes)],
            "coverage": 0,
            "mean_conf": 0.0,
            "mean_entropy": 0.0,
            "mean_proto_dist": 0.0,
            "mean_agree": 0.0,
            "mean_score": 0.0,
            "pred_prior": [0.0 for _ in range(num_classes)],
            "mixed_prior": mixed_prior,
            "quota_per_class": quota_per_class,
            "prior_alpha": prior_alpha,
            "max_prior": 0.0,
            "score_threshold": 0.0,
            "min_class_count": 0,
            "mean_spatial_agree": 0.0,
            "min_spatial_agree": min_spatial_agree,
            "spatial_weight": spatial_weight,
            "spatial_status": "disabled_empty_target_loader",
            "skip_reason": "empty_target_loader",
        }
        info = _attach_generation_info(info)
        info = _attach_candidate_info(
            info,
            empty_dataset,
            _empty_candidate_info(num_classes, candidate_min_conf, candidate_max_entropy, candidate_top_ratio),
            total_count=0,
            core_count=0,
            num_classes=num_classes
        )
        return empty_dataset, info

    data_all = torch.cat(all_data, dim=0)
    pseudo_all = torch.cat(all_pseudo, dim=0).long()
    conf_all = torch.cat(all_conf, dim=0).float()
    entropy_all = torch.cat(all_entropy, dim=0).float()
    proto_dist_all = torch.cat(all_proto_dist, dim=0).float()
    agree_all = torch.cat(all_agree, dim=0).float()
    base_score_all = torch.cat(all_score, dim=0).float()
    gen_rel_all = torch.cat(all_gen_rel, dim=0).float()
    gen_agree_all = torch.cat(all_gen_agree, dim=0).float()
    gen_prob_cons_all = torch.cat(all_gen_prob_cons, dim=0).float()
    gen_feat_cons_all = torch.cat(all_gen_feat_cons, dim=0).float()
    gen_quality_all = torch.cat(all_gen_quality, dim=0).float()

    pred_hist_all = torch.bincount(pseudo_all, minlength=num_classes).float()
    pred_prior = pred_hist_all / pred_hist_all.sum().clamp_min(1.0)
    pred_coverage = int((pred_hist_all > 0).sum().item())
    max_prior = float(pred_prior.max().item()) if pred_prior.numel() > 0 else 0.0
    min_coverage = max(3, math.ceil(num_classes * min_coverage_ratio))

    total_count = data_all.size(0)
    ratio_count = int(total_count * query_ratio)
    num_query = max(batch_size, min(max_query, ratio_count))
    num_query = min(total_count, num_query)
    if min_class_count_abs is None:
        config_matches_call = (
            _LAST_CBP_TDUS_CONFIG.get("query_ratio") == query_ratio
            and _LAST_CBP_TDUS_CONFIG.get("max_query") == max_query
            and _LAST_CBP_TDUS_CONFIG.get("quality_quantile") == quality_quantile
            and _LAST_CBP_TDUS_CONFIG.get("min_selected") == min_selected
            and _LAST_CBP_TDUS_CONFIG.get("min_coverage_ratio") == min_coverage_ratio
            and _LAST_CBP_TDUS_CONFIG.get("max_prior_thr") == max_prior_thr
        )
        if config_matches_call:
            min_class_count_abs = _LAST_CBP_TDUS_CONFIG.get("min_class_count_abs")
    prior_alpha = float(max(0.0, min(1.0, prior_alpha)))
    uniform_prior = torch.ones_like(pred_prior) / num_classes
    mixed_prior = prior_alpha * pred_prior + (1.0 - prior_alpha) * uniform_prior
    mixed_prior = mixed_prior / mixed_prior.sum().clamp_min(1e-8)
    raw_quota = mixed_prior * float(num_query)
    quota_per_class = torch.floor(raw_quota).long()
    remaining_quota = int(num_query - quota_per_class.sum().item())
    if remaining_quota > 0 and min_class_count_abs is None:
        quota_remainder = raw_quota - quota_per_class.float()
        top_remainder = torch.argsort(quota_remainder, descending=True)[:remaining_quota]
        quota_per_class[top_remainder] += 1
    quota_per_class = quota_per_class.clamp_min(0)
    average_quota = max(1, num_query // num_classes)
    if min_class_count_abs is not None:
        min_class_count = int(min_class_count_abs)
    else:
        min_class_count = max(3, int(0.15 * average_quota))

    spatial_agree_all = torch.ones_like(conf_all)
    spatial_status = "disabled"
    has_spatial_coords = (
        target_rows is not None
        and target_cols is not None
        and target_height is not None
        and target_width is not None
    )
    if use_spatial and has_spatial_coords:
        rows = torch.as_tensor(target_rows).reshape(-1).long().cpu()
        cols = torch.as_tensor(target_cols).reshape(-1).long().cpu()
        height = int(target_height)
        width = int(target_width)

        if rows.numel() == total_count and cols.numel() == total_count and height > 0 and width > 0:
            pred_map = torch.full((height, width), -1, dtype=torch.long)
            valid_coord = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
            valid_indices = torch.nonzero(valid_coord, as_tuple=False).view(-1)
            if valid_indices.numel() > 0:
                pred_map[rows[valid_indices], cols[valid_indices]] = pseudo_all[valid_indices]
                radius = max(0, int(spatial_window) // 2)
                for index in valid_indices.tolist():
                    row = int(rows[index].item())
                    col = int(cols[index].item())
                    center_label = int(pseudo_all[index].item())
                    row_start = max(0, row - radius)
                    row_end = min(height, row + radius + 1)
                    col_start = max(0, col - radius)
                    col_end = min(width, col + radius + 1)
                    neighbor = pred_map[row_start:row_end, col_start:col_end]
                    valid_neighbor = neighbor >= 0
                    if valid_neighbor.any():
                        spatial_agree_all[index] = (neighbor[valid_neighbor] == center_label).float().mean()
                spatial_status = "enabled"
            else:
                spatial_status = "disabled_invalid_coordinates"
        else:
            spatial_status = "disabled_invalid_coordinates"
    elif use_spatial:
        spatial_status = "disabled_missing_coordinates"

    base_tdus_score_all = base_score_all + spatial_weight * spatial_agree_all
    active_gen_reliability = bool(use_gen_reliability) and generator_model is not None
    gen_rel_weight = float(max(0.0, min(1.0, gen_rel_weight)))
    gen_rel_mode = str(gen_rel_mode).lower()
    if gen_rel_mode not in ("fusion", "gate_only", "penalty"):
        gen_rel_mode = "gate_only"
    if bool(gen_gate_strict):
        gen_min_agreement = 1.0
    if active_gen_reliability and gen_rel_mode == "fusion":
        base_min = base_tdus_score_all.min()
        base_max = base_tdus_score_all.max()
        if (base_max - base_min).abs() < 1e-8:
            base_score_for_fusion = torch.full_like(base_tdus_score_all, 0.5)
        else:
            base_score_for_fusion = (base_tdus_score_all - base_min) / (base_max - base_min)
        score_all = fuse_tdus_score_with_generation(
            base_score_for_fusion.clamp(0.0, 1.0),
            gen_rel_all.clamp(0.0, 1.0),
            gen_agreement=gen_agree_all.clamp(0.0, 1.0),
            gen_rel_weight=gen_rel_weight,
            use_gen_reliability=True,
            gen_rel_mode=gen_rel_mode,
        )
    elif active_gen_reliability and gen_rel_mode == "penalty":
        score_all = fuse_tdus_score_with_generation(
            base_tdus_score_all,
            gen_rel_all,
            gen_agreement=gen_agree_all,
            gen_rel_weight=gen_rel_weight,
            use_gen_reliability=True,
            gen_rel_mode=gen_rel_mode,
        )
    else:
        score_all = base_tdus_score_all

    mean_spatial_all = float(spatial_agree_all.mean().item()) if spatial_agree_all.numel() > 0 else 0.0
    mean_base_score_all = float(base_tdus_score_all.mean().item()) if base_tdus_score_all.numel() > 0 else 0.0
    mean_final_score_all = float(score_all.mean().item()) if score_all.numel() > 0 else 0.0
    empty_core_mask = torch.zeros(total_count, dtype=torch.bool)
    fallback_candidate_dataset, fallback_candidate_info = _build_candidate_dataset_and_info(
        data_all=data_all,
        pseudo_all=pseudo_all,
        conf_all=conf_all,
        entropy_all=entropy_all,
        score_all=score_all,
        core_mask=empty_core_mask,
        num_classes=num_classes,
        candidate_min_conf=candidate_min_conf,
        candidate_max_entropy=candidate_max_entropy,
        candidate_top_ratio=candidate_top_ratio,
    )

    # Step 1: prediction collapse gate
    # pred_prior 只用于检测类别坍缩，不用于分配 quota
    if pred_coverage < min_coverage or max_prior > max_prior_thr:
        empty_data = data_all[:0].float()
        empty_label = pseudo_all[:0].long()
        empty_weight = score_all[:0].float()

        info = {
            "num_selected": 0,
            "class_hist": [0 for _ in range(num_classes)],
            "core_hist": [0 for _ in range(num_classes)],
            "pre_core_hist": [int(x) for x in pred_hist_all.tolist()],
            "coverage": 0,
            "pre_core_coverage": pred_coverage,
            "mean_conf": 0.0,
            "mean_entropy": 0.0,
            "mean_proto_dist": 0.0,
            "mean_agree": 0.0,
            "mean_score": 0.0,
            "pred_prior": [float(x) for x in pred_prior.tolist()],
            "mixed_prior": [float(x) for x in mixed_prior.tolist()],
            "quota_per_class": [int(x) for x in quota_per_class.tolist()],
            "prior_alpha": prior_alpha,
            "max_prior": max_prior,
            "score_threshold": 0.0,
            "min_class_count": min_class_count,
            "mean_spatial_agree": mean_spatial_all,
            "min_spatial_agree": min_spatial_agree,
            "spatial_weight": spatial_weight,
            "spatial_status": spatial_status,
            "skip_reason": "prediction_collapse",
        }

        info = _attach_generation_info(
            info,
            active_gen_reliability,
            gen_rel_all,
            gen_agree_all,
            gen_prob_cons_all,
            gen_feat_cons_all,
            gen_quality_all,
            base_tdus_score_all,
            score_all,
            gen_rel_weight,
            gen_rel_mode,
            bool(gen_gate_strict),
            gen_min_agreement,
            gen_min_prob_consistency,
            gen_min_quality,
        )
        info = _attach_candidate_info(
            info,
            fallback_candidate_dataset,
            fallback_candidate_info,
            total_count=total_count,
            core_count=0,
            num_classes=num_classes
        )
        return TensorDataset(empty_data, empty_label, empty_weight), info

    # Step 2: first build a basic high-quality candidate mask
    base_candidate_mask = (
            (conf_all >= min_conf)
            & (agree_all >= min_agree)
            & (spatial_agree_all >= min_spatial_agree)
    )

    valid_scores = score_all[base_candidate_mask]

    # Step 3: if no valid candidates, skip this TDUS round
    if valid_scores.numel() == 0:
        empty_data = data_all[:0].float()
        empty_label = pseudo_all[:0].long()
        empty_weight = score_all[:0].float()

        info = {
            "num_selected": 0,
            "class_hist": [0 for _ in range(num_classes)],
            "core_hist": [0 for _ in range(num_classes)],
            "pre_core_hist": [0 for _ in range(num_classes)],
            "coverage": 0,
            "mean_conf": 0.0,
            "mean_entropy": 0.0,
            "mean_proto_dist": 0.0,
            "mean_agree": 0.0,
            "mean_score": 0.0,
            "pred_prior": [float(x) for x in pred_prior.tolist()],
            "mixed_prior": [float(x) for x in mixed_prior.tolist()],
            "quota_per_class": [int(x) for x in quota_per_class.tolist()],
            "prior_alpha": prior_alpha,
            "max_prior": max_prior,
            "score_threshold": 0.0,
            "min_class_count": min_class_count,
            "mean_spatial_agree": mean_spatial_all,
            "min_spatial_agree": min_spatial_agree,
            "spatial_weight": spatial_weight,
            "spatial_status": spatial_status,
            "skip_reason": "no_candidate",
        }

        info = _attach_generation_info(
            info,
            active_gen_reliability,
            gen_rel_all,
            gen_agree_all,
            gen_prob_cons_all,
            gen_feat_cons_all,
            gen_quality_all,
            base_tdus_score_all,
            score_all,
            gen_rel_weight,
            gen_rel_mode,
            bool(gen_gate_strict),
            gen_min_agreement,
            gen_min_prob_consistency,
            gen_min_quality,
        )
        info = _attach_candidate_info(
            info,
            fallback_candidate_dataset,
            fallback_candidate_info,
            total_count=total_count,
            core_count=0,
            num_classes=num_classes
        )
        return TensorDataset(empty_data, empty_label, empty_weight), info

    # Step 4: compute threshold only among valid candidates
    score_threshold = float(torch.quantile(valid_scores, quality_quantile).item())

    core_pool_before_gen_gate = base_candidate_mask & (score_all >= score_threshold)
    effective_gen_min_agreement = 1.0 if bool(gen_gate_strict) else float(gen_min_agreement)
    core_low_agree = int(
        (core_pool_before_gen_gate & (gen_agree_all < effective_gen_min_agreement)).sum().item()
    )
    core_low_prob_cons = int(
        (core_pool_before_gen_gate & (gen_prob_cons_all < float(gen_min_prob_consistency))).sum().item()
    )
    core_low_quality = int(
        (core_pool_before_gen_gate & (gen_quality_all < float(gen_min_quality))).sum().item()
    )
    if active_gen_reliability:
        gate_info = apply_generation_reliability_gate(
            core_mask=core_pool_before_gen_gate,
            candidate_mask=torch.zeros_like(core_pool_before_gen_gate),
            agreement=gen_agree_all,
            prob_consistency=gen_prob_cons_all,
            gen_quality=gen_quality_all,
            gen_min_agreement=effective_gen_min_agreement,
            gen_min_prob_consistency=gen_min_prob_consistency,
            gen_min_quality=gen_min_quality,
        )
        core_pool_mask = gate_info["core_mask"]
        downgraded_to_candidate_mask = gate_info["downgraded_mask"]
    else:
        core_pool_mask = core_pool_before_gen_gate
        downgraded_to_candidate_mask = torch.zeros_like(core_pool_before_gen_gate)
    pre_core_hist = torch.bincount(pseudo_all[core_pool_before_gen_gate], minlength=num_classes).tolist()
    pre_core_coverage = sum([1 for x in pre_core_hist if int(x) > 0])

    selected_mask = torch.zeros(total_count, dtype=torch.bool)

    for class_id in range(num_classes):
        class_indices = torch.nonzero(core_pool_mask & (pseudo_all == class_id), as_tuple=False).view(-1)
        if class_indices.numel() == 0:
            continue
        class_order = torch.argsort(score_all[class_indices], descending=True)
        class_quota = int(quota_per_class[class_id].item())
        if class_quota <= 0:
            continue
        chosen = class_indices[class_order[:class_quota]]
        selected_mask[chosen] = True

    selected_indices = torch.nonzero(selected_mask, as_tuple=False).view(-1)
    if selected_indices.numel() == 0:
        empty_data = data_all[:0].float()
        empty_label = pseudo_all[:0].long()
        empty_weight = score_all[:0].float()
        candidate_dataset, candidate_info = _build_candidate_dataset_and_info(
            data_all=data_all,
            pseudo_all=pseudo_all,
            conf_all=conf_all,
            entropy_all=entropy_all,
            score_all=score_all,
            core_mask=empty_core_mask,
            num_classes=num_classes,
            candidate_min_conf=candidate_min_conf,
            candidate_max_entropy=candidate_max_entropy,
            candidate_top_ratio=candidate_top_ratio,
            extra_candidate_mask=downgraded_to_candidate_mask,
        )
        info = {
            "num_selected": 0,
            "class_hist": [0 for _ in range(num_classes)],
            "core_hist": [0 for _ in range(num_classes)],
            "pre_core_hist": [int(x) for x in pre_core_hist],
            "coverage": 0,
            "pre_core_coverage": pre_core_coverage,
            "mean_conf": 0.0,
            "mean_entropy": 0.0,
            "mean_proto_dist": 0.0,
            "mean_agree": 0.0,
            "mean_score": 0.0,
            "pred_prior": [float(x) for x in pred_prior.tolist()],
            "mixed_prior": [float(x) for x in mixed_prior.tolist()],
            "quota_per_class": [int(x) for x in quota_per_class.tolist()],
            "prior_alpha": prior_alpha,
            "max_prior": max_prior,
            "score_threshold": score_threshold,
            "min_class_count": min_class_count,
            "mean_spatial_agree": mean_spatial_all,
            "min_spatial_agree": min_spatial_agree,
            "spatial_weight": spatial_weight,
            "spatial_status": spatial_status,
            "skip_reason": "no_selected_samples",
        }
        info = _attach_generation_info(
            info,
            active_gen_reliability,
            gen_rel_all,
            gen_agree_all,
            gen_prob_cons_all,
            gen_feat_cons_all,
            gen_quality_all,
            base_tdus_score_all,
            score_all,
            gen_rel_weight,
            gen_rel_mode,
            bool(gen_gate_strict),
            gen_min_agreement,
            gen_min_prob_consistency,
            gen_min_quality,
            core_before_gen_gate=int(core_pool_before_gen_gate.sum().item()),
            core_after_gen_gate=int(core_pool_mask.sum().item()),
            downgraded_to_candidate=int(downgraded_to_candidate_mask.sum().item()),
            core_low_agree=core_low_agree,
            core_low_prob_cons=core_low_prob_cons,
            core_low_quality=core_low_quality,
        )
        info = _attach_candidate_info(
            info,
            candidate_dataset,
            candidate_info,
            total_count=total_count,
            core_count=0,
            num_classes=num_classes
        )
        return TensorDataset(empty_data, empty_label, empty_weight), info

    selected_pseudo = pseudo_all[selected_indices].long()
    class_hist = torch.bincount(selected_pseudo, minlength=num_classes).tolist()
    coverage = sum([1 for x in class_hist if x > 0])
    nonzero_counts = [int(x) for x in class_hist if int(x) > 0]

    if (
        coverage < min_coverage
        or selected_indices.numel() < min_selected
        or len(nonzero_counts) < min_coverage
        or min(nonzero_counts) < min_class_count
    ):
        empty_data = data_all[:0].float()
        empty_label = pseudo_all[:0].long()
        empty_weight = score_all[:0].float()
        if selected_indices.numel() < min_selected:
            skip_reason = "too_few_selected"
        elif coverage < min_coverage:
            skip_reason = "low_selected_coverage"
        else:
            skip_reason = "imbalanced_selected_classes"
        candidate_dataset, candidate_info = _build_candidate_dataset_and_info(
            data_all=data_all,
            pseudo_all=pseudo_all,
            conf_all=conf_all,
            entropy_all=entropy_all,
            score_all=score_all,
            core_mask=empty_core_mask,
            num_classes=num_classes,
            candidate_min_conf=candidate_min_conf,
            candidate_max_entropy=candidate_max_entropy,
            candidate_top_ratio=candidate_top_ratio,
            extra_candidate_mask=(selected_mask | downgraded_to_candidate_mask),
        )
        info = {
            "num_selected": 0,
            "class_hist": [0 for _ in range(num_classes)],
            "core_hist": [0 for _ in range(num_classes)],
            "pre_core_hist": [int(x) for x in class_hist],
            "coverage": 0,
            "pre_core_coverage": coverage,
            "mean_conf": float(conf_all[selected_indices].mean().item()),
            "mean_entropy": float(entropy_all[selected_indices].mean().item()),
            "mean_proto_dist": float(proto_dist_all[selected_indices].mean().item()),
            "mean_agree": float(agree_all[selected_indices].mean().item()),
            "mean_score": float(score_all[selected_indices].mean().item()),
            "pred_prior": [float(x) for x in pred_prior.tolist()],
            "mixed_prior": [float(x) for x in mixed_prior.tolist()],
            "quota_per_class": [int(x) for x in quota_per_class.tolist()],
            "prior_alpha": prior_alpha,
            "max_prior": max_prior,
            "score_threshold": score_threshold,
            "min_class_count": min_class_count,
            "mean_spatial_agree": float(spatial_agree_all[selected_indices].mean().item()),
            "min_spatial_agree": min_spatial_agree,
            "spatial_weight": spatial_weight,
            "spatial_status": spatial_status,
            "skip_reason": skip_reason,
        }
        info = _attach_generation_info(
            info,
            active_gen_reliability,
            gen_rel_all,
            gen_agree_all,
            gen_prob_cons_all,
            gen_feat_cons_all,
            gen_quality_all,
            base_tdus_score_all,
            score_all,
            gen_rel_weight,
            gen_rel_mode,
            bool(gen_gate_strict),
            gen_min_agreement,
            gen_min_prob_consistency,
            gen_min_quality,
            core_before_gen_gate=int(core_pool_before_gen_gate.sum().item()),
            core_after_gen_gate=int(core_pool_mask.sum().item()),
            downgraded_to_candidate=int(downgraded_to_candidate_mask.sum().item()),
            core_low_agree=core_low_agree,
            core_low_prob_cons=core_low_prob_cons,
            core_low_quality=core_low_quality,
        )
        info = _attach_candidate_info(
            info,
            candidate_dataset,
            candidate_info,
            total_count=total_count,
            core_count=0,
            num_classes=num_classes
        )
        return TensorDataset(empty_data, empty_label, empty_weight), info

    selected_scores = score_all[selected_indices]
    score_min = selected_scores.min()
    score_max = selected_scores.max()
    if active_gen_reliability and gen_rel_mode == "fusion":
        normalized_score = selected_scores.clamp(0.0, 1.0)
    elif selected_scores.numel() == 1 or (score_max - score_min).abs() < 1e-8:
        normalized_score = torch.ones_like(selected_scores)
    else:
        normalized_score = (selected_scores - score_min) / (score_max - score_min)
    selected_weight = (
        0.5 * normalized_score.clamp(0.0, 1.0)
        + 0.5 * conf_all[selected_indices].clamp(0.0, 1.0)
    )

    selected_data = data_all[selected_indices].float()
    candidate_dataset, candidate_info = _build_candidate_dataset_and_info(
        data_all=data_all,
        pseudo_all=pseudo_all,
        conf_all=conf_all,
        entropy_all=entropy_all,
        score_all=score_all,
        core_mask=selected_mask,
        num_classes=num_classes,
        candidate_min_conf=candidate_min_conf,
        candidate_max_entropy=candidate_max_entropy,
        candidate_top_ratio=candidate_top_ratio,
        extra_candidate_mask=downgraded_to_candidate_mask,
    )

    info = {
        "num_selected": int(selected_indices.numel()),
        "class_hist": [int(x) for x in class_hist],
        "core_hist": [int(x) for x in class_hist],
        "pre_core_hist": [int(x) for x in pre_core_hist],
        "coverage": coverage,
        "pre_core_coverage": pre_core_coverage,
        "mean_conf": float(conf_all[selected_indices].mean().item()),
        "mean_entropy": float(entropy_all[selected_indices].mean().item()),
        "mean_proto_dist": float(proto_dist_all[selected_indices].mean().item()),
        "mean_agree": float(agree_all[selected_indices].mean().item()),
        "mean_score": float(score_all[selected_indices].mean().item()),
        "pred_prior": [float(x) for x in pred_prior.tolist()],
        "mixed_prior": [float(x) for x in mixed_prior.tolist()],
        "quota_per_class": [int(x) for x in quota_per_class.tolist()],
        "prior_alpha": prior_alpha,
        "max_prior": max_prior,
        "score_threshold": score_threshold,
        "min_class_count": min_class_count,
        "mean_spatial_agree": float(spatial_agree_all[selected_indices].mean().item()),
        "min_spatial_agree": min_spatial_agree,
        "spatial_weight": spatial_weight,
        "spatial_status": spatial_status,
        "skip_reason": "",
    }
    info = _attach_generation_info(
        info,
        active_gen_reliability,
        gen_rel_all,
        gen_agree_all,
        gen_prob_cons_all,
        gen_feat_cons_all,
        gen_quality_all,
        base_tdus_score_all,
        score_all,
        gen_rel_weight,
        gen_rel_mode,
        bool(gen_gate_strict),
        gen_min_agreement,
        gen_min_prob_consistency,
        gen_min_quality,
        core_before_gen_gate=int(core_pool_before_gen_gate.sum().item()),
        core_after_gen_gate=int(core_pool_mask.sum().item()),
        downgraded_to_candidate=int(downgraded_to_candidate_mask.sum().item()),
        core_low_agree=core_low_agree,
        core_low_prob_cons=core_low_prob_cons,
        core_low_quality=core_low_quality,
    )
    info = _attach_candidate_info(
        info,
        candidate_dataset,
        candidate_info,
        total_count=total_count,
        core_count=selected_indices.numel(),
        num_classes=num_classes
    )

    labeled_dataset = TensorDataset(selected_data, selected_pseudo, selected_weight.float())
    return labeled_dataset, info


def weighted_pseudo_label_loss(logits, pseudo_labels, weights):
    ce = F.cross_entropy(logits, pseudo_labels.long(), reduction="none")
    return (ce * weights).mean()
