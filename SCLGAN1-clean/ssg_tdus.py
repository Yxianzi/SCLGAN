import math

import torch
import torch.nn.functional as F


def _feature_vector(x):
    if x is None:
        return None
    if x.dim() > 2:
        return x.reshape(x.size(0), -1)
    return x.reshape(x.size(0), -1)


def _weighted_average(parts):
    total_weight = sum(weight for value, weight in parts if value is not None and weight > 0)
    if total_weight <= 0:
        return None

    out = None
    for value, weight in parts:
        if value is None or weight <= 0:
            continue
        term = value * (weight / total_weight)
        out = term if out is None else out + term
    return out


def js_divergence(p, q, eps=1e-6):
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    p = p / p.sum(dim=1, keepdim=True).clamp_min(eps)
    q = q / q.sum(dim=1, keepdim=True).clamp_min(eps)
    midpoint = 0.5 * (p + q)
    kl_p = torch.sum(p * (torch.log(p) - torch.log(midpoint.clamp_min(eps))), dim=1)
    kl_q = torch.sum(q * (torch.log(q) - torch.log(midpoint.clamp_min(eps))), dim=1)
    return 0.5 * (kl_p + kl_q)


def spectral_angle_mapper(x1, x2, eps=1e-6):
    flat1 = x1.reshape(x1.size(0), -1)
    flat2 = x2.reshape(x2.size(0), -1)
    cos = F.cosine_similarity(flat1, flat2, dim=1, eps=eps)
    cos = cos.clamp(-1.0 + eps, 1.0 - eps)
    return torch.acos(cos) / math.pi


def compute_generation_reliability(
    prob_ori,
    prob_spe,
    prob_spa,
    prob_mix,
    feat_ori=None,
    feat_spe=None,
    feat_spa=None,
    feat_mix=None,
    img_ori=None,
    img_spe=None,
    img_spa=None,
    img_mix=None,
    w_agree=0.35,
    w_prob=0.30,
    w_feat=0.20,
    w_quality=0.15,
    rec_weight=1.0,
    sam_weight=1.0,
    eps=1e-6,
):
    pred_ori = torch.argmax(prob_ori, dim=1)
    pred_spe = torch.argmax(prob_spe, dim=1)
    pred_spa = torch.argmax(prob_spa, dim=1)
    pred_mix = torch.argmax(prob_mix, dim=1)
    agreement = (
        (pred_spe == pred_ori).float()
        + (pred_spa == pred_ori).float()
        + (pred_mix == pred_ori).float()
    ) / 3.0

    js_mean = (
        js_divergence(prob_ori, prob_spe, eps=eps)
        + js_divergence(prob_ori, prob_spa, eps=eps)
        + js_divergence(prob_ori, prob_mix, eps=eps)
    ) / 3.0
    prob_consistency = torch.exp(-js_mean).clamp(0.0, 1.0)

    feat_consistency = None
    feat_ori_vec = _feature_vector(feat_ori)
    feat_spe_vec = _feature_vector(feat_spe)
    feat_spa_vec = _feature_vector(feat_spa)
    feat_mix_vec = _feature_vector(feat_mix)
    if (
        feat_ori_vec is not None
        and feat_spe_vec is not None
        and feat_spa_vec is not None
        and feat_mix_vec is not None
    ):
        cos_spe = F.cosine_similarity(feat_ori_vec, feat_spe_vec, dim=1, eps=eps)
        cos_spa = F.cosine_similarity(feat_ori_vec, feat_spa_vec, dim=1, eps=eps)
        cos_mix = F.cosine_similarity(feat_ori_vec, feat_mix_vec, dim=1, eps=eps)
        feat_consistency = ((cos_spe + cos_spa + cos_mix) / 3.0 + 1.0) / 2.0
        feat_consistency = feat_consistency.clamp(0.0, 1.0)

    rec_mean = None
    sam_mean = None
    gen_quality = None
    if img_ori is not None and img_spe is not None and img_spa is not None and img_mix is not None:
        rec_spe = torch.mean(torch.abs(img_spe - img_ori).reshape(img_ori.size(0), -1), dim=1)
        rec_spa = torch.mean(torch.abs(img_spa - img_ori).reshape(img_ori.size(0), -1), dim=1)
        rec_mix = torch.mean(torch.abs(img_mix - img_ori).reshape(img_ori.size(0), -1), dim=1)
        rec_mean = (rec_spe + rec_spa + rec_mix) / 3.0

        sam_spe = spectral_angle_mapper(img_spe, img_ori, eps=eps)
        sam_spa = spectral_angle_mapper(img_spa, img_ori, eps=eps)
        sam_mix = spectral_angle_mapper(img_mix, img_ori, eps=eps)
        sam_mean = (sam_spe + sam_spa + sam_mix) / 3.0

        gen_shift = float(rec_weight) * rec_mean + float(sam_weight) * sam_mean
        gen_quality = torch.exp(-gen_shift).clamp(0.0, 1.0)

    gen_reliability = _weighted_average([
        (agreement, float(w_agree)),
        (prob_consistency, float(w_prob)),
        (feat_consistency, float(w_feat)),
        (gen_quality, float(w_quality)),
    ])
    if gen_reliability is None:
        gen_reliability = torch.ones_like(agreement)
    gen_reliability = torch.nan_to_num(gen_reliability, nan=0.0, posinf=1.0, neginf=0.0)
    gen_reliability = gen_reliability.clamp(0.0, 1.0)

    return {
        "gen_reliability": gen_reliability,
        "agreement": agreement.clamp(0.0, 1.0),
        "prob_consistency": prob_consistency,
        "feat_consistency": feat_consistency,
        "gen_quality": gen_quality,
        "js_mean": js_mean,
        "sam_mean": sam_mean,
        "rec_mean": rec_mean,
    }


def fuse_tdus_score_with_generation(
    base_tdus_score,
    gen_reliability,
    gen_agreement=None,
    gen_rel_weight=0.30,
    use_gen_reliability=True,
    gen_rel_mode="gate_only",
):
    if not use_gen_reliability:
        return base_tdus_score
    gen_rel_weight = float(max(0.0, min(1.0, gen_rel_weight)))
    if gen_rel_mode == "fusion":
        final_score = (1.0 - gen_rel_weight) * base_tdus_score + gen_rel_weight * gen_reliability
        return final_score.clamp(0.0, 1.0)
    if gen_rel_mode == "penalty":
        if gen_agreement is None:
            gen_agreement = gen_reliability
        return base_tdus_score - gen_rel_weight * (1.0 - gen_agreement)
    return base_tdus_score


def apply_generation_reliability_gate(
    core_mask,
    candidate_mask,
    agreement,
    prob_consistency,
    gen_quality=None,
    gen_min_agreement=0.67,
    gen_min_prob_consistency=0.50,
    gen_min_quality=0.30,
):
    core_mask = core_mask.bool()
    candidate_mask = candidate_mask.bool()
    pass_mask = (
        (agreement >= float(gen_min_agreement))
        & (prob_consistency >= float(gen_min_prob_consistency))
    )
    if gen_quality is not None:
        pass_mask = pass_mask & (gen_quality >= float(gen_min_quality))

    downgraded_mask = core_mask & (~pass_mask)
    new_core_mask = core_mask & pass_mask
    new_candidate_mask = (candidate_mask | downgraded_mask) & (~new_core_mask)
    return {
        "core_mask": new_core_mask,
        "candidate_mask": new_candidate_mask,
        "downgraded_mask": downgraded_mask,
        "num_downgraded": int(downgraded_mask.sum().item()),
    }


def candidate_generation_consistency_loss(
    prob_ori=None,
    prob_spe=None,
    prob_spa=None,
    prob_mix=None,
    feat_ori=None,
    feat_spe=None,
    feat_spa=None,
    feat_mix=None,
    candidate_mask=None,
    loss_type="prob",
    max_raw_loss=2.0,
):
    device_source = prob_ori if prob_ori is not None else feat_ori
    if device_source is None:
        raise ValueError("prob_ori or feat_ori is required")

    if candidate_mask is None:
        candidate_mask = torch.ones(device_source.size(0), dtype=torch.bool, device=device_source.device)
    else:
        candidate_mask = candidate_mask.to(device_source.device).bool()

    num_candidate = int(candidate_mask.sum().item())
    zero_loss = device_source.sum() * 0.0
    if num_candidate == 0:
        return {
            "loss": zero_loss,
            "raw_loss": 0.0,
            "num_candidate": 0,
            "skipped": False,
            "skipped_reason": "",
        }

    losses = []
    if loss_type in ("prob", "both"):
        if prob_ori is None or prob_spe is None or prob_spa is None or prob_mix is None:
            raise ValueError("prob consistency requires all probability inputs")
        target_prob = prob_ori[candidate_mask].detach().clamp_min(1e-8)
        prob_loss = (
            F.kl_div(torch.log(prob_spe[candidate_mask].clamp_min(1e-8)), target_prob, reduction="batchmean")
            + F.kl_div(torch.log(prob_spa[candidate_mask].clamp_min(1e-8)), target_prob, reduction="batchmean")
            + F.kl_div(torch.log(prob_mix[candidate_mask].clamp_min(1e-8)), target_prob, reduction="batchmean")
        ) / 3.0
        losses.append(prob_loss)

    if loss_type in ("feature", "both"):
        if feat_ori is None or feat_spe is None or feat_spa is None or feat_mix is None:
            raise ValueError("feature consistency requires all feature inputs")
        ori = _feature_vector(feat_ori)[candidate_mask].detach()
        spe = _feature_vector(feat_spe)[candidate_mask]
        spa = _feature_vector(feat_spa)[candidate_mask]
        mix = _feature_vector(feat_mix)[candidate_mask]
        feat_loss = (
            (1.0 - F.cosine_similarity(spe, ori, dim=1)).mean()
            + (1.0 - F.cosine_similarity(spa, ori, dim=1)).mean()
            + (1.0 - F.cosine_similarity(mix, ori, dim=1)).mean()
        ) / 3.0
        losses.append(feat_loss)

    if not losses:
        raw_loss = zero_loss
    else:
        raw_loss = sum(losses) / float(len(losses))

    raw_loss_float = float(raw_loss.detach().item())
    if raw_loss_float > float(max_raw_loss):
        return {
            "loss": zero_loss,
            "raw_loss": raw_loss_float,
            "num_candidate": num_candidate,
            "skipped": True,
            "skipped_reason": "raw_loss too high",
        }

    return {
        "loss": raw_loss,
        "raw_loss": raw_loss_float,
        "num_candidate": num_candidate,
        "skipped": False,
        "skipped_reason": "",
    }


def candidate_generation_consistency_from_model(
    model,
    generator_model,
    candidate_data,
    loss_type="prob",
    max_raw_loss=2.0,
):
    if generator_model is None or candidate_data.numel() == 0:
        zero_loss = candidate_data.sum() * 0.0
        return {
            "loss": zero_loss,
            "raw_loss": 0.0,
            "num_candidate": 0,
            "skipped": False,
            "skipped_reason": "",
        }

    generator_was_training = generator_model.training
    generator_model.eval()
    feat_ori, _, _, logits_ori, _ = model(candidate_data)
    with torch.no_grad():
        img_spe, img_spa = generator_model(candidate_data)
        img_mix = (img_spe + img_spa + candidate_data) / 3.0
    if generator_was_training:
        generator_model.train()

    feat_spe, _, _, logits_spe, _ = model(img_spe.detach())
    feat_spa, _, _, logits_spa, _ = model(img_spa.detach())
    feat_mix, _, _, logits_mix, _ = model(img_mix.detach())

    return candidate_generation_consistency_loss(
        prob_ori=F.softmax(logits_ori, dim=1),
        prob_spe=F.softmax(logits_spe, dim=1),
        prob_spa=F.softmax(logits_spa, dim=1),
        prob_mix=F.softmax(logits_mix, dim=1),
        feat_ori=feat_ori,
        feat_spe=feat_spe,
        feat_spa=feat_spa,
        feat_mix=feat_mix,
        candidate_mask=None,
        loss_type=loss_type,
        max_raw_loss=max_raw_loss,
    )
