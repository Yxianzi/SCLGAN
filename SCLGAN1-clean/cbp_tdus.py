import copy
import math

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset


def get_cbp_tdus_config(dataset_name):
    name = dataset_name.lower()

    if name == "houston":
        return {
            "query_ratio": 0.005,
            "max_query": 224,
            "min_conf_start": 0.98,
            "min_conf_floor": 0.90,
            "min_conf_decay": 0.002,
            "min_agree": 1.0,
            "quality_quantile": 0.75,
            "min_selected": 64,
            "min_coverage_ratio": 0.80,
            "max_prior_thr": 0.55,
        }

    if name == "up2pc":
        return {
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
        }

    if name == "sh2hz":
        return {
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
        }

    return {
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
    }


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
    use_spatial=True,
    spatial_window=3,
    min_spatial_agree=0.50,
    spatial_weight=0.20,
    target_rows=None,
    target_cols=None,
    target_height=None,
    target_width=None,
    use_multiview=True,
):
    student_was_training = student_model.training
    student_model.eval()
    teacher_model.eval()

    prototypes_norm = F.normalize(source_prototypes.to(device).float(), dim=1)

    all_data = []
    all_pseudo = []
    all_conf = []
    all_entropy = []
    all_proto_dist = []
    all_agree = []
    all_score = []
    all_topology_gap = []

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

        all_data.append(data.detach().cpu())
        all_pseudo.append(pseudo.detach().cpu())
        all_conf.append(conf.detach().cpu())
        all_entropy.append(entropy.detach().cpu())
        all_proto_dist.append(proto_dist.detach().cpu())
        all_agree.append(agreement.detach().cpu())
        all_score.append(base_score.detach().cpu())
        all_topology_gap.append(topology_gap.detach().cpu())

    if student_was_training:
        student_model.train()
    teacher_model.eval()

    if not all_data:
        empty_data = torch.empty(0)
        empty_label = torch.empty(0, dtype=torch.long)
        empty_weight = torch.empty(0, dtype=torch.float)
        info = {
            "num_selected": 0,
            "class_hist": [0 for _ in range(num_classes)],
            "coverage": 0,
            "mean_conf": 0.0,
            "mean_entropy": 0.0,
            "mean_proto_dist": 0.0,
            "mean_agree": 0.0,
            "mean_score": 0.0,
            "pred_prior": [0.0 for _ in range(num_classes)],
            "max_prior": 0.0,
            "score_threshold": 0.0,
            "min_class_count": 0,
            "mean_spatial_agree": 0.0,
            "min_spatial_agree": min_spatial_agree,
            "spatial_weight": spatial_weight,
            "spatial_status": "disabled_empty_target_loader",
            "skip_reason": "empty_target_loader",
        }
        return TensorDataset(empty_data, empty_label, empty_weight), info

    data_all = torch.cat(all_data, dim=0)
    pseudo_all = torch.cat(all_pseudo, dim=0).long()
    conf_all = torch.cat(all_conf, dim=0).float()
    entropy_all = torch.cat(all_entropy, dim=0).float()
    proto_dist_all = torch.cat(all_proto_dist, dim=0).float()
    agree_all = torch.cat(all_agree, dim=0).float()
    base_score_all = torch.cat(all_score, dim=0).float()

    pred_hist_all = torch.bincount(pseudo_all, minlength=num_classes).float()
    pred_prior = pred_hist_all / pred_hist_all.sum().clamp_min(1.0)
    pred_coverage = int((pred_hist_all > 0).sum().item())
    max_prior = float(pred_prior.max().item()) if pred_prior.numel() > 0 else 0.0
    min_coverage = max(3, math.ceil(num_classes * min_coverage_ratio))

    total_count = data_all.size(0)
    ratio_count = int(total_count * query_ratio)
    num_query = max(batch_size, min(max_query, ratio_count))
    num_query = min(total_count, num_query)
    quota = max(1, num_query // num_classes)
    min_class_count = max(3, int(0.15 * quota))

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

    score_all = base_score_all + spatial_weight * spatial_agree_all

    mean_spatial_all = float(spatial_agree_all.mean().item()) if spatial_agree_all.numel() > 0 else 0.0

    # Step 1: prediction collapse gate
    # pred_prior 只用于检测类别坍缩，不用于分配 quota
    if pred_coverage < min_coverage or max_prior > max_prior_thr:
        empty_data = data_all[:0].float()
        empty_label = pseudo_all[:0].long()
        empty_weight = score_all[:0].float()

        info = {
            "num_selected": 0,
            "class_hist": [int(x) for x in pred_hist_all.tolist()],
            "coverage": pred_coverage,
            "mean_conf": 0.0,
            "mean_entropy": 0.0,
            "mean_proto_dist": 0.0,
            "mean_agree": 0.0,
            "mean_score": 0.0,
            "pred_prior": [float(x) for x in pred_prior.tolist()],
            "max_prior": max_prior,
            "score_threshold": 0.0,
            "min_class_count": min_class_count,
            "mean_spatial_agree": mean_spatial_all,
            "min_spatial_agree": min_spatial_agree,
            "spatial_weight": spatial_weight,
            "spatial_status": spatial_status,
            "skip_reason": "prediction_collapse",
        }

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
            "coverage": 0,
            "mean_conf": 0.0,
            "mean_entropy": 0.0,
            "mean_proto_dist": 0.0,
            "mean_agree": 0.0,
            "mean_score": 0.0,
            "pred_prior": [float(x) for x in pred_prior.tolist()],
            "max_prior": max_prior,
            "score_threshold": 0.0,
            "min_class_count": min_class_count,
            "mean_spatial_agree": mean_spatial_all,
            "min_spatial_agree": min_spatial_agree,
            "spatial_weight": spatial_weight,
            "spatial_status": spatial_status,
            "skip_reason": "no_candidate",
        }

        return TensorDataset(empty_data, empty_label, empty_weight), info

    # Step 4: compute threshold only among valid candidates
    score_threshold = float(torch.quantile(valid_scores, quality_quantile).item())

    candidate_mask = base_candidate_mask & (score_all >= score_threshold)

    selected_mask = torch.zeros(total_count, dtype=torch.bool)

    for class_id in range(num_classes):
        class_indices = torch.nonzero(candidate_mask & (pseudo_all == class_id), as_tuple=False).view(-1)
        if class_indices.numel() == 0:
            continue
        class_order = torch.argsort(score_all[class_indices], descending=True)
        chosen = class_indices[class_order[:quota]]
        selected_mask[chosen] = True

    selected_indices = torch.nonzero(selected_mask, as_tuple=False).view(-1)
    if selected_indices.numel() == 0:
        empty_data = data_all[:0].float()
        empty_label = pseudo_all[:0].long()
        empty_weight = score_all[:0].float()
        info = {
            "num_selected": 0,
            "class_hist": [0 for _ in range(num_classes)],
            "coverage": 0,
            "mean_conf": 0.0,
            "mean_entropy": 0.0,
            "mean_proto_dist": 0.0,
            "mean_agree": 0.0,
            "mean_score": 0.0,
            "pred_prior": [float(x) for x in pred_prior.tolist()],
            "max_prior": max_prior,
            "score_threshold": score_threshold,
            "min_class_count": min_class_count,
            "mean_spatial_agree": mean_spatial_all,
            "min_spatial_agree": min_spatial_agree,
            "spatial_weight": spatial_weight,
            "spatial_status": spatial_status,
            "skip_reason": "no_selected_samples",
        }
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
        info = {
            "num_selected": 0,
            "class_hist": [int(x) for x in class_hist],
            "coverage": coverage,
            "mean_conf": float(conf_all[selected_indices].mean().item()),
            "mean_entropy": float(entropy_all[selected_indices].mean().item()),
            "mean_proto_dist": float(proto_dist_all[selected_indices].mean().item()),
            "mean_agree": float(agree_all[selected_indices].mean().item()),
            "mean_score": float(score_all[selected_indices].mean().item()),
            "pred_prior": [float(x) for x in pred_prior.tolist()],
            "max_prior": max_prior,
            "score_threshold": score_threshold,
            "min_class_count": min_class_count,
            "mean_spatial_agree": float(spatial_agree_all[selected_indices].mean().item()),
            "min_spatial_agree": min_spatial_agree,
            "spatial_weight": spatial_weight,
            "spatial_status": spatial_status,
            "skip_reason": skip_reason,
        }
        return TensorDataset(empty_data, empty_label, empty_weight), info

    selected_scores = score_all[selected_indices]
    score_min = selected_scores.min()
    score_max = selected_scores.max()
    if selected_scores.numel() == 1 or (score_max - score_min).abs() < 1e-8:
        normalized_score = torch.ones_like(selected_scores)
    else:
        normalized_score = (selected_scores - score_min) / (score_max - score_min)
    selected_weight = 0.2 + 0.8 * normalized_score.clamp(0.0, 1.0)

    selected_data = data_all[selected_indices].float()

    info = {
        "num_selected": int(selected_indices.numel()),
        "class_hist": [int(x) for x in class_hist],
        "coverage": coverage,
        "mean_conf": float(conf_all[selected_indices].mean().item()),
        "mean_entropy": float(entropy_all[selected_indices].mean().item()),
        "mean_proto_dist": float(proto_dist_all[selected_indices].mean().item()),
        "mean_agree": float(agree_all[selected_indices].mean().item()),
        "mean_score": float(score_all[selected_indices].mean().item()),
        "pred_prior": [float(x) for x in pred_prior.tolist()],
        "max_prior": max_prior,
        "score_threshold": score_threshold,
        "min_class_count": min_class_count,
        "mean_spatial_agree": float(spatial_agree_all[selected_indices].mean().item()),
        "min_spatial_agree": min_spatial_agree,
        "spatial_weight": spatial_weight,
        "spatial_status": spatial_status,
        "skip_reason": "",
    }

    labeled_dataset = TensorDataset(selected_data, selected_pseudo, selected_weight.float())
    return labeled_dataset, info


def weighted_pseudo_label_loss(logits, pseudo_labels, weights):
    ce = F.cross_entropy(logits, pseudo_labels.long(), reduction="none")
    return (ce * weights).mean()
