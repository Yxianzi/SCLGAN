import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import mmd
import numpy as np
from sklearn import metrics

import time
import utils
from utils import sample_gt
from torch.utils.data import TensorDataset, DataLoader
from contrastive_loss import SupConLoss
from config_UP2PC import *
from sklearn import svm
import matplotlib.pyplot as plt
from generator import SSDGnet, CSSGnet
from Discriminator import discriminator
from cbp_tdus import (
    ModelEMA,
    compute_source_prototypes,
    build_cbp_tdus_dataset,
    weighted_pseudo_label_loss,
    strong_augment_torch,
    get_cbp_tdus_config,
)
from ssg_tdus import candidate_generation_consistency_from_model
from losses.tgccal import tdus_guided_ccal_loss
import argparse

import warnings
warnings.filterwarnings('ignore')
##################################

parser = argparse.ArgumentParser(description='PyTorch KI_GAN')

group_model = parser.add_argument_group('model')
group_model.add_argument('--pro_dim', type=int, default=128)
group_model.add_argument("--GIN", type=bool, default=True, help='global intensity non-linear augmentation')
group_model.add_argument("--adv", type=bool, default=True, help='global intensity non-linear augmentation')
group_model.add_argument("--noise", type=bool, default=True, help='noise z')
group_model.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
group_model.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
group_model.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
group_model.add_argument('--GIN_ch', type=int, default=24, help='channel of GIN')
group_model.add_argument('--n_bands', type=int, default=102)
group_model.add_argument('--gen_type', type=str, default='cssg', choices=['original', 'cssg'])
group_model.add_argument('--gen_gamma', type=float, default=0.05)
group_model.add_argument('--gen_preserve_weight', type=float, default=0.01)
group_model.add_argument('--gen_sam_weight', type=float, default=0.1)
group_model.add_argument('--gen_lr_ratio', type=float, default=0.1,
                         help='learning rate ratio for generator relative to feature encoder')
group_model.add_argument('--use_tdus', action='store_true', default=False, help='enable TDUS pseudo-label training')
group_model.add_argument('--align_type', type=str, default='none',
                         choices=['none', 'global_cal', 'tg_ccal'],
                         help='alignment loss type')
group_model.add_argument('--lambda_con_s', type=float, default=1.0)
group_model.add_argument('--lambda_con_t', type=float, default=1.0)
group_model.add_argument('--tgccal_weight', type=float, default=0.01, help='TG-CCAL loss weight')
group_model.add_argument('--tgccal_min_samples', type=int, default=2,
                         help='minimum samples per class for TG-CCAL')
group_model.add_argument('--tgccal_use_cov', type=int, default=1, help='use covariance in TG-CCAL')
group_model.add_argument('--tgccal_use_mean', type=int, default=1, help='use mean in TG-CCAL')
group_model.add_argument('--tgccal_use_proto', type=int, default=1, help='use prototype cosine in TG-CCAL')
group_model.add_argument('--min_required_selected', type=int, default=CLASS_NUM * 4,
                         help='minimum core samples for full TG-CCAL weight')
group_model.add_argument('--min_required_coverage', type=int, default=5)
group_model.add_argument('--min_valid_classes', type=int, default=5)
group_model.add_argument('--candidate_min_conf', type=float, default=0.80)
group_model.add_argument('--candidate_max_entropy', type=float, default=0.80)
group_model.add_argument('--candidate_top_ratio', type=float, default=0.20)
group_model.add_argument('--use_candidate_consistency', type=int, default=1)
group_model.add_argument('--candidate_consistency_weight', type=float, default=0.001)
group_model.add_argument('--max_candidate_kl_loss', type=float, default=2.0)
group_model.add_argument('--use_candidate_curriculum', type=int, default=1)
group_model.add_argument('--candidate_conf_start', type=float, default=0.90)
group_model.add_argument('--candidate_conf_end', type=float, default=0.80)
group_model.add_argument('--candidate_entropy_start', type=float, default=0.50)
group_model.add_argument('--candidate_entropy_end', type=float, default=0.80)
group_model.add_argument('--use_gen_reliability', type=int, default=1)
group_model.add_argument('--gen_rel_weight', type=float, default=0.30)
group_model.add_argument('--gen_rel_mode', type=str, default='gate_only',
                         choices=['fusion', 'gate_only', 'penalty'])
group_model.add_argument('--gen_rel_rec_weight', type=float, default=1.0)
group_model.add_argument('--gen_rel_sam_weight', type=float, default=1.0)
group_model.add_argument('--gen_rel_w_agree', type=float, default=0.35)
group_model.add_argument('--gen_rel_w_prob', type=float, default=0.30)
group_model.add_argument('--gen_rel_w_feat', type=float, default=0.20)
group_model.add_argument('--gen_rel_w_quality', type=float, default=0.15)
group_model.add_argument('--gen_min_agreement', type=float, default=0.67)
group_model.add_argument('--gen_min_prob_consistency', type=float, default=0.50)
group_model.add_argument('--gen_min_quality', type=float, default=0.30)
group_model.add_argument('--gen_gate_strict', type=int, default=0)
group_model.add_argument('--use_candidate_gen_consistency', type=int, default=1)
group_model.add_argument('--candidate_gen_cons_type', type=str, default='prob',
                         choices=['prob', 'feature', 'both'])
group_model.add_argument('--lambda_candidate_gen_cons', type=float, default=0.001)
group_model.add_argument('--candidate_gen_cons_max_loss', type=float, default=2.0)
group_model.add_argument('--use_reliability_tce', type=int, default=1)
group_model.add_argument('--lambda_tgt_ce', type=float, default=0.05)
group_model.add_argument('--verbose_tdus_log', type=int, default=1)

args = parser.parse_args()
print(args)

data_path_s = './datasets/Pavia/paviaU.mat'
label_path_s = './datasets/Pavia/paviaU_gt_7.mat'
data_path_t = './datasets/Pavia/pavia.mat'
label_path_t = './datasets/Pavia/pavia_gt_7.mat'

data_s,label_s = utils.load_data_pavia(data_path_s,label_path_s)
data_t,label_t = utils.load_data_pavia(data_path_t,label_path_t)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_name = "UP2pc"
USE_TDUS = args.use_tdus
USE_DBR = False
USE_CPC = False
USE_PPA = False
USE_MCC = False
USE_BNM = False

# Loss Function
crossEntropy = nn.CrossEntropyLoss().cuda()
ContrastiveLoss_s = SupConLoss(temperature=0.1).cuda()
ContrastiveLoss_t = SupConLoss(temperature=0.1).cuda()
DSH_loss = utils.Domain_Occ_loss().cuda()

lambda_ent = 0.0
lambda_bal = 0.0
lambda_cpc = 0.0
lambda_cpc_cov = 0.0
cpc_warmup_epoch = 30
cpc_ramp_epoch = 20


def candidate_thresholds_for_epoch(epoch):
    if not bool(args.use_candidate_curriculum):
        return args.candidate_min_conf, args.candidate_max_entropy
    progress = min(max((epoch - train_num) / float(max(epochs - train_num, 1)), 0.0), 1.0)
    candidate_min_conf = args.candidate_conf_start + progress * (args.candidate_conf_end - args.candidate_conf_start)
    candidate_max_entropy = (
        args.candidate_entropy_start
        + progress * (args.candidate_entropy_end - args.candidate_entropy_start)
    )
    return candidate_min_conf, candidate_max_entropy


acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, CLASS_NUM])
k = np.zeros([nDataSet, 1])
best_predict_all = []
best_acc_all = 0.0
best_G,best_RandPerm,best_Row, best_Column,best_nTrain = None,None,None,None,None
def correlation_alignment_loss(source_features, target_features):
    # 计算源域和目标域的特征的相关性矩阵
    source_cov = torch.cov(source_features.t())
    target_cov = torch.cov(target_features.t())
    # 计算相关对齐损失
    loss = torch.mean((source_cov - target_cov) ** 2)
    return loss


def distribution_balance_regularization(target_outputs, num_classes, lambda_ent=0.0, lambda_bal=0.0):
    target_prob = torch.softmax(target_outputs, dim=1)
    entropy_loss = -torch.mean(torch.sum(target_prob * torch.log(target_prob + 1e-8), dim=1))
    p_mean = target_prob.mean(dim=0)
    uniform = torch.ones_like(p_mean) / num_classes
    balance_loss = torch.sum(p_mean * torch.log((p_mean + 1e-8) / (uniform + 1e-8)))
    loss_dbr = lambda_ent * entropy_loss + lambda_bal * balance_loss
    return loss_dbr, entropy_loss, balance_loss, target_prob


def sam_loss(x, y, eps=1e-6):
    x_flat = x.reshape(x.size(0), -1)
    y_flat = y.reshape(y.size(0), -1)
    cos = F.cosine_similarity(x_flat, y_flat, dim=1, eps=eps)
    cos = cos.clamp(-1.0 + eps, 1.0 - eps)
    return torch.acos(cos).mean()


def class_prototype_covariance_alignment(
        source_features,
        source_labels,
        target_features,
        target_prob,
        num_classes,
        lambda_cov=0.0):
    loss_proto = source_features.new_tensor(0.0)
    loss_cov = source_features.new_tensor(0.0)
    valid_classes = 0
    source_labels = source_labels.view(-1).long()
    target_prob = target_prob.detach()

    for class_id in range(num_classes):
        source_mask = source_labels == class_id
        source_count = source_mask.sum()
        if source_count.item() < 2:
            continue

        target_weight = target_prob[:, class_id]
        target_weight_sum = target_weight.sum()
        if target_weight_sum.detach().item() < 1e-3:
            continue

        class_source_features = source_features[source_mask]
        source_proto = class_source_features.mean(dim=0)
        target_proto = (target_features * target_weight.unsqueeze(1)).sum(dim=0) / target_weight_sum.clamp_min(1e-8)

        source_centered = class_source_features - source_proto
        target_centered = target_features - target_proto
        source_cov = source_centered.t().mm(source_centered) / (source_count.float() - 1.0)
        target_cov = (target_centered * target_weight.unsqueeze(1)).t().mm(target_centered) / target_weight_sum.clamp_min(1e-8)

        loss_proto = loss_proto + torch.sum((source_proto - target_proto) ** 2)
        loss_cov = loss_cov + torch.sum((source_cov - target_cov) ** 2)
        valid_classes += 1

    if valid_classes == 0:
        return source_features.new_tensor(0.0)
    return loss_proto + lambda_cov * loss_cov


for iDataSet in range(nDataSet):
    print('#######################idataset######################## ', iDataSet)

    utils.set_seed(seeds[iDataSet])

    trainX, trainY = utils.get_sample_data(data_s, label_s, HalfWidth, 180)
    testID, testX, testY, G, RandPerm, Row, Column = utils.get_all_data(data_t, label_t, HalfWidth)

    train_dataset = TensorDataset(torch.tensor(trainX), torch.tensor(trainY))
    test_dataset = TensorDataset(torch.tensor(testX), torch.tensor(testY))

    train_loader_s = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    train_loader_t = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)
    test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False,drop_last=False)

    len_source_loader = len(train_loader_s)
    len_target_loader = len(train_loader_t)

    # model
    feature_encoder =discriminator(nBand, 128,  CLASS_NUM,patch_size).cuda()
    teacher_encoder = ModelEMA(feature_encoder, decay=0.99)
    if args.gen_type == 'cssg':
        G_net = CSSGnet(args).cuda()
    else:
        G_net = SSDGnet(args).cuda()
    optimizer = torch.optim.AdamW([
        {
            "params": feature_encoder.parameters(),
            "lr": lr,
            "weight_decay": l2_decay,
            "name": "feature_encoder",
        },
        {
            "params": G_net.parameters(),
            "lr": lr * args.gen_lr_ratio,
            "weight_decay": l2_decay,
            "name": "generator",
        },
    ], betas=(0.9, 0.999), eps=1e-8)

    print("Training...")

    last_accuracy = 0.0
    best_episdoe = 0
    best_aa = 0.0
    best_kappa = 0.0
    train_loss = []
    test_acc = []
    running_D_loss, running_F_loss = 0.0, 0.0
    running_label_loss = 0
    running_domain_loss = 0
    total_hit, total_num = 0.0, 0.0
    size = 0.0
    test_acc_list = []
    cls_losses = []
    lmmd_losses = []
    contrastive_losses_s = []
    contrastive_losses_t = []
    domain_similar_losses = []
    total_losses = []
    loss_min_losses = []
    loss_correlation_alignment = []
    ACC = []
    train_start = time.time()
    #loss plot
    loss1 = []
    loss2 = []
    loss3 = []
    labeled_loader = None
    candidate_loader = None
    tdus_info = {
        "num_core": 0,
        "num_candidate": 0,
        "num_unselected": 0,
        "pre_core_hist": [0 for _ in range(CLASS_NUM)],
        "core_hist": [0 for _ in range(CLASS_NUM)],
        "candidate_hist": [0 for _ in range(CLASS_NUM)],
        "missing_core_classes": [class_id for class_id in range(CLASS_NUM)],
        "missing_candidate_classes": [class_id for class_id in range(CLASS_NUM)],
        "coverage": 0,
    }

    for epoch in range(1, epochs + 1):
        LEARNING_RATE = lr / math.pow((1 + 10 * (epoch - 1) / epochs), 0.75)
        for param_group in optimizer.param_groups:
            if param_group.get("name", "") == "generator":
                param_group["lr"] = LEARNING_RATE * args.gen_lr_ratio
            else:
                param_group["lr"] = LEARNING_RATE
        print('learning rate feature={:.6f}, generator={:.6f}'.format(
            LEARNING_RATE, LEARNING_RATE * args.gen_lr_ratio
        ))
        tdus_refreshed = False
        cls_criterion = torch.nn.CrossEntropyLoss()

        feature_encoder.train()
        G_net.train()
        iter_source = iter(train_loader_s)
        iter_target = iter(train_loader_t)

        if USE_TDUS and (epoch >= train_num and epoch < epochs) and ((epoch - train_num) % 20 == 0):
            source_prototypes = compute_source_prototypes(
                feature_encoder, train_loader, CLASS_NUM, device
            )

            cbp_cfg = get_cbp_tdus_config(dataset_name)
            candidate_min_conf, candidate_max_entropy = candidate_thresholds_for_epoch(epoch)

            min_conf = max(
                cbp_cfg["min_conf_floor"],
                cbp_cfg["min_conf_start"] - cbp_cfg["min_conf_decay"] * max(0, epoch - train_num)
            )

            labeled_dataset, tdus_info = build_cbp_tdus_dataset(
                student_model=feature_encoder,
                teacher_model=teacher_encoder.ema,
                target_loader=test_loader,
                source_prototypes=source_prototypes,
                num_classes=CLASS_NUM,
                batch_size=BATCH_SIZE,
                device=device,
                query_ratio=cbp_cfg["query_ratio"],
                max_query=cbp_cfg["max_query"],
                min_conf=min_conf,
                min_agree=cbp_cfg["min_agree"],
                quality_quantile=cbp_cfg["quality_quantile"],
                min_selected=cbp_cfg["min_selected"],
                min_coverage_ratio=cbp_cfg["min_coverage_ratio"],
                max_prior_thr=cbp_cfg["max_prior_thr"],
                prior_alpha=cbp_cfg["prior_alpha"],
                use_spatial=True,
                spatial_window=3,
                min_spatial_agree=0.50,
                spatial_weight=0.05,
                target_rows=Row[RandPerm],
                target_cols=Column[RandPerm],
                target_height=G.shape[0],
                target_width=G.shape[1],
                use_multiview=True,
                candidate_min_conf=candidate_min_conf,
                candidate_max_entropy=candidate_max_entropy,
                candidate_top_ratio=args.candidate_top_ratio,
                generator_model=G_net,
                use_gen_reliability=bool(args.use_gen_reliability),
                gen_rel_weight=args.gen_rel_weight,
                gen_rel_mode=args.gen_rel_mode,
                gen_gate_strict=bool(args.gen_gate_strict),
                gen_rel_rec_weight=args.gen_rel_rec_weight,
                gen_rel_sam_weight=args.gen_rel_sam_weight,
                gen_rel_w_agree=args.gen_rel_w_agree,
                gen_rel_w_prob=args.gen_rel_w_prob,
                gen_rel_w_feat=args.gen_rel_w_feat,
                gen_rel_w_quality=args.gen_rel_w_quality,
                gen_min_agreement=args.gen_min_agreement,
                gen_min_prob_consistency=args.gen_min_prob_consistency,
                gen_min_quality=args.gen_min_quality,
            )
            tdus_refreshed = True

            if bool(args.verbose_tdus_log) and tdus_info.get("spatial_status", "") in ("disabled_missing_coordinates", "disabled_invalid_coordinates"):
                print("[SC-DAPT-TDUS] spatial disabled due to missing coordinates:", tdus_info["spatial_status"])

            if tdus_info["num_selected"] == 0:
                labeled_loader = None
                if bool(args.verbose_tdus_log):
                    print(
                        "[SC-DAPT-TDUS] Skip pseudo-label fine-tuning: "
                        f"reason={tdus_info['skip_reason']}, "
                        f"core_hist={tdus_info['class_hist']}, "
                        f"pre_core_hist={tdus_info.get('pre_core_hist', [])}, "
                        f"coverage={tdus_info['coverage']}, "
                        f"max_prior={tdus_info['max_prior']:.4f}, "
                        f"mixed_prior={['{:.3f}'.format(x) for x in tdus_info['mixed_prior']]}, "
                        f"quota={tdus_info['quota_per_class']}, "
                        f"score_threshold={tdus_info['score_threshold']:.4f}, "
                        f"min_class_count={tdus_info['min_class_count']}, "
                        f"spa_agree={tdus_info['mean_spatial_agree']:.4f}"
                    )
            else:
                labeled_loader = DataLoader(
                    labeled_dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    drop_last=True
                )
            candidate_dataset = tdus_info.get("candidate_dataset", None)
            if (
                (bool(args.use_candidate_consistency) or bool(args.use_candidate_gen_consistency))
                and candidate_dataset is not None
                and len(candidate_dataset) > 0
            ):
                candidate_loader = DataLoader(
                    candidate_dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    drop_last=False
                )
            else:
                candidate_loader = None

            if bool(args.verbose_tdus_log):
                print(
                    "[SC-DAPT-TDUS][{}] epoch={}, selected={}, core_hist={}, pre_core_hist={}, coverage={}, pred_prior={}, mixed_prior={}, quota_per_class={}, max_prior={:.4f}, score_threshold={:.4f}, min_class_count={}, spa_agree={:.4f}, skip_reason={}, conf={:.4f}, ent={:.4f}, proto_dist={:.4f}, agree={:.4f}, score={:.4f}".format(
                        dataset_name,
                        epoch,
                        tdus_info["num_selected"],
                        tdus_info["class_hist"],
                        tdus_info.get("pre_core_hist", []),
                        tdus_info["coverage"],
                        ["{:.3f}".format(x) for x in tdus_info["pred_prior"]],
                        ["{:.3f}".format(x) for x in tdus_info["mixed_prior"]],
                        tdus_info["quota_per_class"],
                        tdus_info["max_prior"],
                        tdus_info["score_threshold"],
                        tdus_info["min_class_count"],
                        tdus_info["mean_spatial_agree"],
                        tdus_info["skip_reason"],
                        tdus_info["mean_conf"],
                        tdus_info["mean_entropy"],
                        tdus_info["mean_proto_dist"],
                        tdus_info["mean_agree"],
                        tdus_info["mean_score"],
                    )
                )
            # print("开始主动学习")
            # 评估未标注数据的预测概率

                    # 选择不确定性最高的样本进行标注
                    # 计算每个样本的熵
                    # 获取新标注的测试数据

        feature_encoder.train()
        # G_net.train()
        use_cbp_tdus = (
            USE_TDUS
            and epoch >= train_num
            and "labeled_loader" in locals()
            and labeled_loader is not None
            and len(labeled_loader) > 0
        )
        if use_cbp_tdus:
            iter_update = iter(labeled_loader)
            len_update_loader = len(labeled_loader)
        use_candidate_consistency = (
            USE_TDUS
            and (bool(args.use_candidate_consistency) or bool(args.use_candidate_gen_consistency))
            and epoch >= train_num
            and candidate_loader is not None
            and len(candidate_loader) > 0
        )
        if use_candidate_consistency:
            iter_candidate = iter(candidate_loader)
            len_candidate_loader = len(candidate_loader)

        num_iter = len_source_loader
        epoch_dbr_ent = 0.0
        epoch_dbr_bal = 0.0
        epoch_cpc_cal_loss = 0.0
        epoch_cpc_weight = 0.0
        epoch_loss_steps = 0
        tgccal_epoch_info = {
            "valid_classes": 0,
            "selected": 0,
            "loss": 0.0,
            "mean": 0.0,
            "cov": 0.0,
            "proto": 0.0,
            "class_counts": [0 for _ in range(CLASS_NUM)],
            "skipped_classes": [class_id for class_id in range(CLASS_NUM)],
            "core_global": 0,
            "core_batch": 0,
            "effective_ratio": 0.0,
            "weighted_loss": 0.0,
            "skip_reason": "not_computed",
        }
        candidate_epoch_loss = 0.0
        candidate_epoch_count = 0
        candidate_epoch_steps = 0
        candidate_batch_last = 0
        candidate_skip_epoch = False
        candidate_skip_reason = ""
        candidate_gen_epoch_loss = 0.0
        candidate_gen_epoch_weighted = 0.0
        candidate_gen_epoch_count = 0
        candidate_gen_epoch_steps = 0
        candidate_gen_skip_reason = ""
        tgt_ce_epoch_raw = 0.0
        tgt_ce_epoch_weighted = 0.0
        tgt_ce_epoch_count = 0
        tgt_ce_epoch_steps = 0

        for i in range(1,num_iter):
            source_data01, source_label = next(iter_source)
            try:
                target_data01, _ = next(iter_target)
            except StopIteration:
                iter_target = iter(train_loader_t)
                target_data01, _ = next(iter_target)
            source_data00 = source_data01.cuda()
            source_label = source_label.cuda()

            target_data00 = target_data01.cuda()


            if i % len_target_loader == 0:
                iter_target = iter(train_loader_t)

            source_aug_img1, source_aug_img2 = G_net(source_data00)
            target_aug_img1, target_aug_img2 = G_net(target_data00)
            loss_gen_rec = (
                F.l1_loss(source_aug_img1, source_data00)
                + F.l1_loss(source_aug_img2, source_data00)
                + F.l1_loss(target_aug_img1, target_data00)
                + F.l1_loss(target_aug_img2, target_data00)
            )
            loss_gen_sam = (
                sam_loss(source_aug_img1, source_data00)
                + sam_loss(source_aug_img2, source_data00)
                + sam_loss(target_aug_img1, target_data00)
                + sam_loss(target_aug_img2, target_data00)
            )
            loss_gen_preserve = loss_gen_rec + args.gen_sam_weight * loss_gen_sam
            alpha1, alpha2, alpha3 = np.random.dirichlet([0.6, 0.6, 0.6])
            source_data = alpha1 * source_aug_img1 + alpha2 * source_aug_img2 + alpha3 * source_data00
            target_data = alpha1 * target_aug_img1 + alpha2 * target_aug_img2 + alpha3 * target_data00

            _, _, _, predict1, _ = feature_encoder(source_aug_img1.detach())
            _, _, _, predict2, _ = feature_encoder(source_aug_img2.detach())
            _, _, _, predict3, _ = feature_encoder(source_data.detach())

            _, _, _, predict4, _ = feature_encoder(target_aug_img1.detach())
            _, _, _, predict5, _ = feature_encoder(target_aug_img2.detach())
            _, _, _, predict6, _ = feature_encoder(target_data.detach())

            loss_aug1 = cls_criterion(predict1, source_label.long())
            loss_aug2 = cls_criterion(predict2, source_label.long())
            loss_aug3 = cls_criterion(predict3, source_label.long())

            loss_aug4 = torch.zeros((), device=target_data00.device)
            loss_aug5 = torch.zeros((), device=target_data00.device)
            loss_aug6 = torch.zeros((), device=target_data00.device)

            prob1 = torch.softmax(predict1, dim=1)
            prob2 = torch.softmax(predict2, dim=1)
            prob3 = torch.softmax(predict3, dim=1)

            prob4 = torch.softmax(predict4, dim=1)
            prob5 = torch.softmax(predict5, dim=1)
            prob6 = torch.softmax(predict6, dim=1)

            source_features, source1, _, source_outputs, source_out = feature_encoder(source_data01.cuda())
            _, source2, _, _, _ = feature_encoder(source_aug_img1.cuda())
            _, source3, _, _, _ = feature_encoder(source_aug_img2.cuda())

            target_features, target1, _, target_outputs, target_out = feature_encoder(target_data01.cuda())
            _, _, target2, t1, _ = feature_encoder(target_aug_img1.cuda())
            _, _, target3, t2, _ = feature_encoder(target_aug_img2.cuda())

            if USE_DBR:
                loss_dbr, dbr_ent, dbr_bal, target_prob = distribution_balance_regularization(
                    target_outputs,
                    CLASS_NUM,
                    lambda_ent=lambda_ent,
                    lambda_bal=lambda_bal
                )
            else:
                target_prob = torch.softmax(target_outputs, dim=1)
                loss_dbr = target_outputs.new_tensor(0.0)
                dbr_ent = target_outputs.new_tensor(0.0)
                dbr_bal = target_outputs.new_tensor(0.0)
            softmax_output_t = target_prob.detach()
            _, pseudo_label_t = torch.max(softmax_output_t, 1)
            tdus_clean_batch = None
            clean_tgccal_feat = None
            if use_cbp_tdus and args.align_type == "tg_ccal":
                if i % len_update_loader == 0:
                    iter_update = iter(labeled_loader)
                try:
                    clean_data, clean_label, clean_weight = next(iter_update)
                except StopIteration:
                    iter_update = iter(labeled_loader)
                    clean_data, clean_label, clean_weight = next(iter_update)
                clean_data = clean_data.cuda().float()
                clean_label = clean_label.cuda().long()
                clean_weight = clean_weight.cuda().float()
                tdus_clean_batch = (clean_data, clean_label, clean_weight)
                _, clean_tgccal_feat, _, _, _ = feature_encoder(clean_data)

            # Supervised Contrastive Loss
            all_source_con_features = torch.cat([source2.unsqueeze(1), source3.unsqueeze(1)], dim=1)  # 32  2 128
            # print("all_source_con_features",all_source_con_features.shape)
            all_target_con_features = torch.cat([target2.unsqueeze(1), target3.unsqueeze(1)], dim=1)  # 32 2 128
            # print("all_target_con_features",all_target_con_features.shape)
            lmmd_loss = mmd.lmmd(source_features, target_features, source_label,
                                 target_prob, BATCH_SIZE=BATCH_SIZE,
                                 CLASS_NUM=CLASS_NUM)
            lambd = 2 / (1 + math.exp(-10 * (epoch) / epochs)) - 1
            # 计算源域和目标域之间的距离
            # distances = torch.norm(source1 - target1, p=2, dim=1)
            # 计算损失
            # loss_yuan = torch.mean(distances)

            # Loss Cls
            cls_loss = crossEntropy(source_outputs, source_label.cuda())
            # print("cls_loss", cls_loss)

            # Loss Con_s

            contrastive_loss_s = ContrastiveLoss_s(all_source_con_features, source_label)  #

            # Loss Con_t
            contrastive_loss_t = ContrastiveLoss_t(all_target_con_features, pseudo_label_t)  #
            # print(contrastive_loss_t)
            tgccal_info = {
                "valid_classes": 0,
                "selected": 0,
                "loss": 0.0,
                "mean": 0.0,
                "cov": 0.0,
                "proto": 0.0,
                "class_counts": [0 for _ in range(CLASS_NUM)],
                "skipped_classes": [class_id for class_id in range(CLASS_NUM)],
                "core_global": 0,
                "core_batch": 0,
                "effective_ratio": 0.0,
                "weighted_loss": 0.0,
                "skip_reason": "not_computed",
            }
            if args.align_type == "none":
                loss_correlation_alignment_loss = source1.sum() * 0.0
            elif args.align_type == "global_cal":
                loss_correlation_alignment_loss = correlation_alignment_loss(source1, target1)
            else:
                if tdus_clean_batch is not None and clean_tgccal_feat is not None:
                    _, clean_label_for_align, clean_weight_for_align = tdus_clean_batch
                    core_selected_count = int(tdus_info.get("num_core", tdus_info.get("num_selected", 0)))
                    core_coverage = int(tdus_info.get("coverage", 0))
                    core_batch_count = int(clean_label_for_align.numel())
                    tgccal_info["core_global"] = core_selected_count
                    tgccal_info["core_batch"] = core_batch_count
                    if (
                        core_selected_count < int(args.min_required_selected)
                        or core_coverage < int(args.min_required_coverage)
                    ):
                        loss_correlation_alignment_loss = source1.sum() * 0.0
                        tgccal_info["skip_reason"] = "insufficient_core_or_coverage"
                    else:
                        tgccal_raw_loss, tgccal_info = tdus_guided_ccal_loss(
                            source_feat=source1,
                            source_y=source_label,
                            target_feat=clean_tgccal_feat,
                            target_pseudo_y=clean_label_for_align,
                            target_conf=clean_weight_for_align,
                            selected_mask=None,
                            num_classes=CLASS_NUM,
                            min_samples_per_class=args.tgccal_min_samples,
                            use_cov=bool(args.tgccal_use_cov),
                            use_mean=bool(args.tgccal_use_mean),
                            use_proto=bool(args.tgccal_use_proto),
                            return_details=True,
                        )
                        tgccal_info["core_global"] = core_selected_count
                        tgccal_info["core_batch"] = core_batch_count
                        if tgccal_info["valid_classes"] < int(args.min_valid_classes):
                            loss_correlation_alignment_loss = source1.sum() * 0.0
                            tgccal_info["effective_ratio"] = 0.0
                            tgccal_info["weighted_loss"] = 0.0
                            tgccal_info["skip_reason"] = "insufficient_core_or_coverage"
                        else:
                            valid_class_ratio = min(1.0, tgccal_info["valid_classes"] / float(max(1, CLASS_NUM)))
                            effective_ratio = valid_class_ratio
                            loss_correlation_alignment_loss = args.tgccal_weight * effective_ratio * tgccal_raw_loss
                            tgccal_info["effective_ratio"] = effective_ratio
                            tgccal_info["weighted_loss"] = float(loss_correlation_alignment_loss.detach().item())
                            tgccal_info["skip_reason"] = ""
                else:
                    loss_correlation_alignment_loss = source1.sum() * 0.0
                    tgccal_info["core_global"] = int(tdus_info.get("num_core", tdus_info.get("num_selected", 0)))
                    tgccal_info["skip_reason"] = "insufficient_core_or_coverage"
            tgccal_epoch_info = tgccal_info
            if USE_CPC or USE_PPA:
                cpc_cal_loss = class_prototype_covariance_alignment(
                    source_features,
                    source_label,
                    target_features,
                    target_prob.detach(),
                    CLASS_NUM,
                    lambda_cov=lambda_cpc_cov
                )
            else:
                cpc_cal_loss = target_outputs.new_tensor(0.0)

            # loss_center_alignment_loss = center_alignment_loss(source1, target1)

            # Loss Occ
            domain_similar_loss = DSH_loss(source_out, target_out)  #
            # print(domain_similar_loss)

            source_loss_kl = F.kl_div(
                F.log_softmax(predict1, dim=1),
                F.softmax(predict3.detach(), dim=1),
                reduction="batchmean"
            ) + F.kl_div(
                F.log_softmax(predict2, dim=1),
                F.softmax(predict3.detach(), dim=1),
                reduction="batchmean"
            )

            target_loss_kl = F.kl_div(
                F.log_softmax(predict4, dim=1),
                F.softmax(predict6.detach(), dim=1),
                reduction="batchmean"
            ) + F.kl_div(
                F.log_softmax(predict5, dim=1),
                F.softmax(predict6.detach(), dim=1),
                reduction="batchmean"
            )

            loss_min = loss_aug1 + loss_aug2 + loss_aug3 + source_loss_kl + 0.5 * target_loss_kl
            # print("loss_min", loss_min)

            cpc_progress = min(max((epoch - cpc_warmup_epoch) / float(cpc_ramp_epoch), 0.0), 1.0)
            cpc_weight = lambda_cpc * cpc_progress if (USE_CPC or USE_PPA) else 0.0
            weighted_con_s = args.lambda_con_s * contrastive_loss_s
            weighted_con_t = args.lambda_con_t * contrastive_loss_t
            weighted_gen_preserve = args.gen_preserve_weight * loss_gen_preserve
            loss = (
                cls_loss
                + 0.01 * lambd * lmmd_loss
                + weighted_con_s
                + weighted_con_t
                + domain_similar_loss
                + loss_correlation_alignment_loss
                + loss_min
                + weighted_gen_preserve
            )
            # print(loss)
            epoch_dbr_ent += dbr_ent.item()
            epoch_dbr_bal += dbr_bal.item()
            epoch_cpc_cal_loss += cpc_cal_loss.item()
            epoch_cpc_weight += cpc_weight
            epoch_loss_steps += 1

            pred = source_outputs.data.max(1)[1]
            total_hit += pred.eq(source_label.data).sum()
            size += source_label.data.size()[0]
            test_accuracy = 100. * float(total_hit) / size
            test_loss = None
            if use_cbp_tdus or use_candidate_consistency:
                if use_cbp_tdus:

                    if tdus_clean_batch is None:
                        if i % len_update_loader == 0:
                            iter_update = iter(labeled_loader)
                        try:
                            clean_data, clean_label, clean_weight = next(iter_update)
                        except StopIteration:
                            iter_update = iter(labeled_loader)
                            clean_data, clean_label, clean_weight = next(iter_update)
                        clean_data = clean_data.cuda().float()
                        clean_label = clean_label.cuda().long()
                        clean_weight = clean_weight.cuda().float()
                    else:
                        clean_data, clean_label, clean_weight = tdus_clean_batch

                    try:
                        target_data_test, _ = next(iter_target)
                    except StopIteration:
                        iter_target = iter(train_loader_t)
                        target_data_test, _ = next(iter_target)
                    target_data_test = target_data_test.cuda().float()

                    _, _, _, clean_outputs, _ = feature_encoder(clean_data)
                    loss_pseudo = weighted_pseudo_label_loss(
                        clean_outputs,
                        clean_label,
                        clean_weight
                    )

                    with torch.no_grad():
                        _, _, _, teacher_logits, _ = teacher_encoder.ema(target_data_test)
                        teacher_prob = F.softmax(teacher_logits, dim=1)

                    target_strong = strong_augment_torch(target_data_test)

                    _, _, _, student_logits, _ = feature_encoder(target_strong)

                    loss_cons = F.kl_div(
                        F.log_softmax(student_logits, dim=1),
                        teacher_prob.detach(),
                        reduction="batchmean"
                    )

                    tgt_ce_weight = args.lambda_tgt_ce if bool(args.use_reliability_tce) else 0.1
                    tgt_ce_weighted = tgt_ce_weight * loss_pseudo
                    tgt_ce_epoch_raw += loss_pseudo.item()
                    tgt_ce_epoch_weighted += tgt_ce_weighted.item()
                    tgt_ce_epoch_count += clean_data.size(0)
                    tgt_ce_epoch_steps += 1
                    test_loss = tgt_ce_weighted + 0.1 * loss_cons

                if use_candidate_consistency and not candidate_skip_epoch:
                    if i % len_candidate_loader == 0:
                        iter_candidate = iter(candidate_loader)
                    try:
                        candidate_data, _, _ = next(iter_candidate)
                    except StopIteration:
                        iter_candidate = iter(candidate_loader)
                        candidate_data, _, _ = next(iter_candidate)
                    candidate_data = candidate_data.cuda().float()
                    candidate_batch_last = candidate_data.size(0)

                    if bool(args.use_candidate_consistency):
                        with torch.no_grad():
                            _, _, _, candidate_teacher_logits, _ = teacher_encoder.ema(candidate_data)
                            candidate_teacher_prob = F.softmax(candidate_teacher_logits, dim=1)

                        candidate_strong = strong_augment_torch(candidate_data)
                        _, _, _, candidate_student_logits, _ = feature_encoder(candidate_strong)
                        loss_candidate = F.kl_div(
                            F.log_softmax(candidate_student_logits, dim=1),
                            candidate_teacher_prob.detach(),
                            reduction="batchmean"
                        )
                        if loss_candidate.detach().item() > float(args.max_candidate_kl_loss):
                            candidate_skip_epoch = True
                            candidate_skip_reason = "raw_loss too high"
                        else:
                            candidate_epoch_loss += loss_candidate.item()
                            candidate_epoch_count += candidate_data.size(0)
                            candidate_epoch_steps += 1
                            candidate_weighted_loss = args.candidate_consistency_weight * loss_candidate
                            test_loss = candidate_weighted_loss if test_loss is None else test_loss + candidate_weighted_loss

                    if bool(args.use_candidate_gen_consistency):
                        candidate_gen_result = candidate_generation_consistency_from_model(
                            feature_encoder,
                            G_net,
                            candidate_data,
                            loss_type=args.candidate_gen_cons_type,
                            max_raw_loss=args.candidate_gen_cons_max_loss,
                        )
                        if candidate_gen_result["skipped"]:
                            candidate_gen_skip_reason = candidate_gen_result["skipped_reason"]
                        else:
                            loss_candidate_gen = candidate_gen_result["loss"]
                            candidate_gen_weighted = args.lambda_candidate_gen_cons * loss_candidate_gen
                            candidate_gen_epoch_loss += candidate_gen_result["raw_loss"]
                            candidate_gen_epoch_weighted += candidate_gen_weighted.item()
                            candidate_gen_epoch_count += candidate_gen_result["num_candidate"]
                            candidate_gen_epoch_steps += 1
                            test_loss = candidate_gen_weighted if test_loss is None else test_loss + candidate_gen_weighted

            total_batch_loss = loss + test_loss if test_loss is not None else loss
            optimizer.zero_grad()
            total_batch_loss.backward()
            optimizer.step()
            teacher_encoder.update(feature_encoder)

        print(
            'epoch {:>3d}:   cls loss: {:6.4f}, lmmd loss:{:6.4f}, occ loss:{:6f}, con_s loss:{:6f}, con_t loss:{:6f}, acc {:6.4f}, loss_min:{:6.4f}, cal_loss:{:6.4f}, total loss: {:6.4f}'
                .format(epoch, cls_loss.item(), lmmd_loss.item(), domain_similar_loss.item(), contrastive_loss_s.item(),
                        contrastive_loss_t.item(),
                        total_hit / size, loss_min.item(), loss_correlation_alignment_loss.item(), loss.item()))
        print(
            "[GEN-PRESERVE] type={}, gamma={:.4f}, rec={:.6f}, sam={:.6f}, preserve={:.6f}, weighted={:.6f}".format(
                args.gen_type,
                args.gen_gamma,
                loss_gen_rec.item(),
                loss_gen_sam.item(),
                loss_gen_preserve.item(),
                weighted_gen_preserve.item(),
            )
        )
        with torch.no_grad():
            delta_spe = torch.mean(torch.abs(source_aug_img1 - source_data00)).item()
            delta_spa = torch.mean(torch.abs(source_aug_img2 - source_data00)).item()
        print("[GEN-DELTA] spe={:.6f}, spa={:.6f}".format(delta_spe, delta_spa))
        tdus_detail_print = print if (tdus_refreshed and bool(args.verbose_tdus_log)) else (lambda *args, **kwargs: None)
        tdus_detail_print(
            "[CON] raw_s={:.6f}, raw_t={:.6f}, lambda_s={:.4f}, lambda_t={:.4f}, weighted_s={:.6f}, weighted_t={:.6f}".format(
                contrastive_loss_s.item(),
                contrastive_loss_t.item(),
                args.lambda_con_s,
                args.lambda_con_t,
                weighted_con_s.item(),
                weighted_con_t.item(),
            )
        )
        tdus_detail_print(
            "[GEN-REL] mean={:.4f}, min={:.4f}, max={:.4f}, agree_mean={:.4f}, prob_cons_mean={:.4f}, feat_cons_mean={:.4f}, gen_quality_mean={:.4f}".format(
                tdus_info.get("gen_rel_mean", 0.0),
                tdus_info.get("gen_rel_min", 0.0),
                tdus_info.get("gen_rel_max", 0.0),
                tdus_info.get("gen_agree_mean", 0.0),
                tdus_info.get("gen_prob_cons_mean", 0.0),
                tdus_info.get("gen_feat_cons_mean", 0.0),
                tdus_info.get("gen_quality_mean", 0.0),
            )
        )
        tdus_detail_print(
            "[GEN-REL] low_agree={}, low_prob_cons={}, low_quality={}".format(
                tdus_info.get("low_agree", 0),
                tdus_info.get("low_prob_cons", 0),
                tdus_info.get("low_quality", 0),
            )
        )
        tdus_detail_print(
            "[TDUS] base_score_mean={:.4f}, final_score_mean={:.4f}, gen_rel_weight={:.4f}".format(
                tdus_info.get("base_score_mean", 0.0),
                tdus_info.get("final_score_mean", 0.0),
                tdus_info.get("gen_rel_weight", 0.0),
            )
        )
        tdus_detail_print(
            "[GEN-GATE] core_pool_global={}, core_low_agree_global={}, core_low_prob_cons_global={}, core_low_quality_global={}, downgraded_global={}".format(
                tdus_info.get("core_before_gen_gate", 0),
                tdus_info.get("core_low_agree", 0),
                tdus_info.get("core_low_prob_cons", 0),
                tdus_info.get("core_low_quality", 0),
                tdus_info.get("downgraded_to_candidate", 0),
            )
        )
        tdus_detail_print(
            "[TDUS] core_before_gen_gate={}, core_after_gen_gate={}, downgraded_to_candidate={}".format(
                tdus_info.get("core_before_gen_gate", 0),
                tdus_info.get("core_after_gen_gate", 0),
                tdus_info.get("downgraded_to_candidate", 0),
            )
        )
        tdus_detail_print(
            "[ETDUS] core_global={}, candidate_global={}, unselected={}, pre_core_hist={}, core_hist={}, candidate_hist={}, coverage={}, missing_core_classes={}, missing_candidate_classes={}".format(
                tdus_info.get("num_core", tdus_info.get("num_selected", 0)),
                tdus_info.get("num_candidate", 0),
                tdus_info.get("num_unselected", 0),
                tdus_info.get("pre_core_hist", [0 for _ in range(CLASS_NUM)]),
                tdus_info.get("core_hist", tdus_info.get("class_hist", [0 for _ in range(CLASS_NUM)])),
                tdus_info.get("candidate_hist", [0 for _ in range(CLASS_NUM)]),
                tdus_info.get("coverage", 0),
                tdus_info.get("missing_core_classes", []),
                tdus_info.get("missing_candidate_classes", []),
            )
        )
        if args.align_type == "tg_ccal":
            tdus_detail_print(
                "[TG-CCAL] epoch={}, core_global={}, core_batch={}, effective_ratio={:.4f}, valid_classes={}, skipped_classes={}, loss={:.6f}, raw_loss={:.6f}, mean={:.6f}, cov={:.6f}, proto={:.6f}, class_counts_batch={}".format(
                    epoch,
                    tgccal_epoch_info.get("core_global", 0),
                    tgccal_epoch_info.get("core_batch", 0),
                    tgccal_epoch_info.get("effective_ratio", 0.0),
                    tgccal_epoch_info["valid_classes"],
                    tgccal_epoch_info.get("skipped_classes", []),
                    tgccal_epoch_info.get("weighted_loss", 0.0),
                    tgccal_epoch_info.get("loss", 0.0),
                    tgccal_epoch_info["mean"],
                    tgccal_epoch_info["cov"],
                    tgccal_epoch_info["proto"],
                    tgccal_epoch_info["class_counts"],
                )
            )
            if tgccal_epoch_info.get("skip_reason", ""):
                tdus_detail_print("[TG-CCAL][SKIP] reason={}".format(tgccal_epoch_info["skip_reason"]))
        candidate_avg_loss = candidate_epoch_loss / max(1, candidate_epoch_steps)
        tdus_detail_print(
            "[CAND-KL] candidate_global={}, candidate_used={}, candidate_batch={}, loss={:.6f}, weight={:.6f}".format(
                tdus_info.get("num_candidate", 0),
                candidate_epoch_count,
                candidate_batch_last,
                candidate_avg_loss,
                args.candidate_consistency_weight if bool(args.use_candidate_consistency) else 0.0,
            )
        )
        if candidate_skip_epoch:
            tdus_detail_print("[CAND-KL][SKIP] {}".format(candidate_skip_reason))
        tgt_ce_avg_raw = tgt_ce_epoch_raw / max(1, tgt_ce_epoch_steps)
        tgt_ce_avg_weighted = tgt_ce_epoch_weighted / max(1, tgt_ce_epoch_steps)
        tdus_detail_print(
            "[TGT-CE] core={}, raw={:.6f}, weighted={:.6f}".format(
                tgt_ce_epoch_count,
                tgt_ce_avg_raw,
                tgt_ce_avg_weighted,
            )
        )
        candidate_gen_avg_loss = candidate_gen_epoch_loss / max(1, candidate_gen_epoch_steps)
        candidate_gen_avg_weighted = candidate_gen_epoch_weighted / max(1, candidate_gen_epoch_steps)
        tdus_detail_print(
            "[CAND-GEN-CONS] candidate={}, raw={:.6f}, weighted={:.6f}, skipped_reason={}".format(
                candidate_gen_epoch_count,
                candidate_gen_avg_loss,
                candidate_gen_avg_weighted,
                candidate_gen_skip_reason,
            )
        )

        cls_losses.append(cls_loss.item())

        lmmd_losses.append(lmmd_loss.item())
        contrastive_losses_s.append(contrastive_loss_s.item())
        contrastive_losses_t.append(contrastive_loss_t.item())
        domain_similar_losses.append(domain_similar_loss.item())
        loss_min_losses.append(loss_min.item())
        loss_correlation_alignment.append(loss_correlation_alignment_loss.item())
        total_losses.append(loss.item())
        ACC.append((total_hit / size).item())

        if epoch == epochs:
            # print(ACC)
            plt.plot(ACC, label='ACC')
            plt.xlabel('Epoch')
            plt.ylabel('ACC')
            plt.title('ACC value')
            plt.legend()
            plt.pause(0.01)
            plt.savefig('acc.png')
            plt.show()
        if epoch == epochs:
            plt.plot(total_losses, label='total_loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('pavia Loss Curve')
            plt.legend()
            plt.pause(0.01)
            plt.savefig('loss_curve.png')
            plt.show()
        train_end = time.time()
        if epoch % 10 == 0 or epoch == epochs:
            print("Testing ...")
            feature_encoder.eval()
            total_rewards = 0
            counter = 0
            accuracies = []
            predict = np.array([], dtype=np.int64)
            labels = np.array([], dtype=np.int64)
            with torch.no_grad():
                for test_datas, test_labels in test_loader:
                    batch_size = test_labels.shape[0]

                    test_features, _, _, test_outputs, _ = feature_encoder(Variable(test_datas).cuda())

                    pred = test_outputs.data.max(1)[1]

                    test_labels = test_labels.numpy()
                    rewards = [1 if pred[j] == test_labels[j] else 0 for j in range(batch_size)]

                    total_rewards += np.sum(rewards)
                    counter += batch_size

                    predict = np.append(predict, pred.cpu().numpy())
                    labels = np.append(labels, test_labels)

                    accuracy = total_rewards / 1.0 / counter  #
                    accuracies.append(accuracy)

            test_accuracy = 100. * total_rewards / len(test_loader.dataset)
            C = metrics.confusion_matrix(labels, predict)
            class_accuracy = np.diag(C) / np.sum(C, 1, dtype=float)
            eval_aa = np.mean(class_accuracy)
            eval_kappa = metrics.cohen_kappa_score(labels, predict)

            print(
                "[Eval][{}] seed_id={}, epoch={}, OA={:.4f}, AA={:.4f}, Kappa={:.4f}".format(
                    dataset_name,
                    iDataSet,
                    epoch,
                    test_accuracy,
                    100. * eval_aa,
                    100. * eval_kappa,
                )
            )
            print('\t\tAccuracy: {}/{} ({:.2f}%)\n'.format(total_rewards, len(test_loader.dataset),
                                                           100. * total_rewards / len(test_loader.dataset)))
            if last_accuracy > 0 and test_accuracy < last_accuracy - 3.0:
                print("[WARN] target OA degraded from best by more than 3%.")
            test_end = time.time()

            # Training mode

            if test_accuracy > last_accuracy:
                # save networks
                # torch.save(feature_encoder.state_dict(),str("../checkpoints/DFSL_feature_encoder_" + "houston_cl_lmmd_dis_attention" +str(iDataSet) +".pkl"))
                print("save networks for epoch:", epoch)
                last_accuracy = test_accuracy
                best_episdoe = epoch
                best_aa = eval_aa
                best_kappa = eval_kappa
                acc[iDataSet] = test_accuracy
                A[iDataSet, :] = class_accuracy
                k[iDataSet] = eval_kappa
                best_predict_all = predict
                best_G, best_RandPerm, best_Row, best_Column = G, RandPerm, Row, Column
                print(
                    "[Best][{}] seed_id={}, best_epoch={}, best_OA={:.4f}, best_AA={:.4f}, best_Kappa={:.4f}".format(
                        dataset_name,
                        iDataSet,
                        best_episdoe,
                        last_accuracy,
                        100. * best_aa,
                        100. * best_kappa,
                    )
                )

            print('iter:{} best epoch:[{}], best accuracy={}'.format(iDataSet, best_episdoe, last_accuracy))
            if epoch == epochs:
                print(
                    "[Best][{}] seed_id={}, best_epoch={}, best_OA={:.4f}, best_AA={:.4f}, best_Kappa={:.4f}".format(
                        dataset_name,
                        iDataSet,
                        best_episdoe,
                        last_accuracy,
                        100. * best_aa,
                        100. * best_kappa,
                    )
                )
            print('***********************************************************************************')

AA = np.mean(A, 1)
AAMean = np.mean(AA,0)
AAStd = np.std(AA)
AMean = np.mean(A, 0)
AStd = np.std(A, 0)
OAMean = np.mean(acc)
OAStd = np.std(acc)
kMean = np.mean(k)
kStd = np.std(k)
print ("train time per DataSet(s): " + "{:.5f}".format(train_end-train_start))
print("test time per DataSet(s): " + "{:.5f}".format(test_end-train_end))
print("average best OA: " + "{:.2f}".format(OAMean))
print("average best AA: " + "{:.2f}".format(100 * AAMean))
print("average best Kappa: " + "{:.4f}".format(100 * kMean))
print("std OA: " + "{:.2f}".format(OAStd))
print("std AA: " + "{:.2f}".format(100 * AAStd))
print("std Kappa: " + "{:.4f}".format(100 * kStd))
print ("accuracy for each class: ")
for i in range(CLASS_NUM):
    print ("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))

best_iDataset = 0
for i in range(len(acc)):
    print('{}:{}'.format(i, acc[i]))
    if acc[i] > acc[best_iDataset]:
        best_iDataset = i
print('best acc all={}'.format(acc[best_iDataset]))

#################classification map################################

for i in range(len(best_predict_all)):  # predict ndarray <class 'tuple'>: (9729,)
    best_G[best_Row[best_RandPerm[ i]]][best_Column[best_RandPerm[ i]]] = best_predict_all[i] + 1

hsi_pic = np.zeros((best_G.shape[0], best_G.shape[1], 3))
for i in range(best_G.shape[0]):
    for j in range(best_G.shape[1]):
        if best_G[i][j] == 0:
            hsi_pic[i, j, :] = [0, 0, 0]
        if best_G[i][j] == 1:
            hsi_pic[i, j, :] = [0, 0, 1]
        if best_G[i][j] == 2:
            hsi_pic[i, j, :] = [0, 1, 0]
        if best_G[i][j] == 3:
            hsi_pic[i, j, :] = [0, 1, 1]
        if best_G[i][j] == 4:
            hsi_pic[i, j, :] = [1, 0, 0]
        if best_G[i][j] == 5:
            hsi_pic[i, j, :] = [1, 0, 1]
        if best_G[i][j] == 6:
            hsi_pic[i, j, :] = [1, 1, 0]
        if best_G[i][j] == 7:
            hsi_pic[i, j, :] = [0.5, 0.5, 1]

    utils.classification_map(hsi_pic[4:-4, 4:-4, :], best_G[4:-4, 4:-4], 24,  "/tmp/pycharm_project_566/pavia_map")
