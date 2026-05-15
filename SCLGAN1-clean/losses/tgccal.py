import torch
import torch.nn.functional as F


def _as_2d_features(features):
    if features.dim() == 2:
        return features
    if features.dim() > 2:
        return features.mean(dim=tuple(range(2, features.dim())))
    return features.view(features.size(0), -1)


def _covariance(features, eps):
    num_samples = features.size(0)
    centered = features - features.mean(dim=0, keepdim=True)
    cov = centered.t().matmul(centered) / max(num_samples - 1, 1)
    eye = torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
    return cov + eps * eye


def _empty_info(num_classes):
    return {
        "valid_classes": 0,
        "selected": 0,
        "loss": 0.0,
        "mean": 0.0,
        "cov": 0.0,
        "proto": 0.0,
        "class_counts": [0 for _ in range(num_classes or 0)],
        "skipped_classes": [class_id for class_id in range(num_classes or 0)],
    }


def tdus_guided_ccal_loss(
        source_feat,
        source_y,
        target_feat,
        target_pseudo_y,
        target_conf=None,
        selected_mask=None,
        num_classes=None,
        min_samples_per_class=2,
        use_cov=True,
        use_mean=True,
        use_proto=True,
        eps=1e-6,
        return_details=False):
    """TDUS-guided class-conditional correlation alignment.

    Target samples are expected to be TDUS-selected reliable samples. If a
    selected_mask is provided, it is applied before per-class alignment.
    """
    if num_classes is None:
        num_classes = int(source_y.max().item()) + 1

    zero_loss = source_feat.sum() * 0.0
    if target_feat is None or target_pseudo_y is None or target_pseudo_y.numel() == 0:
        info = _empty_info(num_classes)
        return (zero_loss, info) if return_details else zero_loss

    source_feat = _as_2d_features(source_feat)
    target_feat = _as_2d_features(target_feat)
    source_y = source_y.view(-1).long().to(source_feat.device)
    target_pseudo_y = target_pseudo_y.view(-1).long().to(target_feat.device)

    target_count = target_feat.size(0)
    if selected_mask is None:
        selected_mask = torch.ones(target_count, dtype=torch.bool, device=target_feat.device)
    else:
        selected_mask = selected_mask.view(-1).bool().to(target_feat.device)

    if target_conf is not None:
        target_conf = target_conf.view(-1).float().to(target_feat.device)

    if selected_mask.numel() != target_count:
        raise ValueError("selected_mask must have the same length as target_feat")

    selected_total = int(selected_mask.sum().item())
    class_counts = [0 for _ in range(num_classes)]
    if selected_total == 0:
        info = _empty_info(num_classes)
        return (zero_loss, info) if return_details else zero_loss

    total_loss = source_feat.new_tensor(0.0)
    total_weight = source_feat.new_tensor(0.0)
    mean_acc = source_feat.new_tensor(0.0)
    cov_acc = source_feat.new_tensor(0.0)
    proto_acc = source_feat.new_tensor(0.0)
    valid_classes = 0
    skipped_classes = []

    for class_id in range(num_classes):
        source_mask = source_y == class_id
        target_mask = (target_pseudo_y == class_id) & selected_mask

        source_k = source_feat[source_mask]
        target_k = target_feat[target_mask]
        class_counts[class_id] = int(target_mask.sum().item())

        if source_k.size(0) < min_samples_per_class or target_k.size(0) < min_samples_per_class:
            skipped_classes.append(class_id)
            continue

        source_mean = source_k.mean(dim=0)
        target_mean = target_k.mean(dim=0)

        mean_loss = source_feat.new_tensor(0.0)
        cov_loss = source_feat.new_tensor(0.0)
        proto_loss = source_feat.new_tensor(0.0)

        if use_mean:
            mean_loss = F.mse_loss(source_mean, target_mean, reduction="sum")

        if use_cov:
            source_cov = _covariance(source_k, eps)
            target_cov = _covariance(target_k, eps)
            feat_dim = max(source_feat.size(1), 1)
            cov_loss = torch.sum((source_cov - target_cov) ** 2) / float(feat_dim * feat_dim)

        if use_proto:
            proto_loss = 1.0 - F.cosine_similarity(
                source_mean.unsqueeze(0),
                target_mean.unsqueeze(0),
                dim=1,
                eps=eps
            ).mean()

        if target_conf is not None:
            class_weight = target_conf[target_mask].mean().clamp_min(eps)
        else:
            class_weight = source_feat.new_tensor(1.0)

        class_loss = mean_loss + cov_loss + proto_loss
        total_loss = total_loss + class_weight * class_loss
        total_weight = total_weight + class_weight
        mean_acc = mean_acc + class_weight * mean_loss
        cov_acc = cov_acc + class_weight * cov_loss
        proto_acc = proto_acc + class_weight * proto_loss
        valid_classes += 1

    if valid_classes == 0 or total_weight.item() <= 0:
        info = _empty_info(num_classes)
        info["selected"] = selected_total
        info["class_counts"] = class_counts
        return (zero_loss, info) if return_details else zero_loss

    loss = total_loss / total_weight
    if not torch.isfinite(loss):
        info = _empty_info(num_classes)
        info["selected"] = selected_total
        info["class_counts"] = class_counts
        return (zero_loss, info) if return_details else zero_loss

    info = {
        "valid_classes": valid_classes,
        "selected": selected_total,
        "loss": float(loss.detach().item()),
        "mean": float((mean_acc / total_weight).detach().item()),
        "cov": float((cov_acc / total_weight).detach().item()),
        "proto": float((proto_acc / total_weight).detach().item()),
        "class_counts": class_counts,
        "skipped_classes": skipped_classes,
    }
    return (loss, info) if return_details else loss
