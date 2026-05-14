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
from generator import SSDGnet
from Discriminator import discriminator
from cbp_tdus import (
    ModelEMA,
    compute_source_prototypes,
    build_cbp_tdus_dataset,
    weighted_pseudo_label_loss,
    strong_augment_torch,
    get_cbp_tdus_config,
)
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
USE_TDUS = False
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
    D_opt = torch.optim.Adam(feature_encoder.parameters())
    G_net = SSDGnet(args).cuda()
    G_opt = torch.optim.Adam(G_net.parameters())

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

    for epoch in range(1, epochs + 1):
        LEARNING_RATE = lr / math.pow((1 + 10 * (epoch - 1) / epochs), 0.75)
        print('learning rate{: .4f}'.format(LEARNING_RATE))
        optimizer = torch.optim.SGD([
            {'params': feature_encoder.mp.parameters(), 'lr': LEARNING_RATE},
            {'params': feature_encoder.Spatial_Weight_1.parameters(), 'lr': LEARNING_RATE},
            {'params': feature_encoder.Spectral_Weight_2.parameters(), 'lr': LEARNING_RATE},
            {'params': feature_encoder.Spectral_Weight_3.parameters(), 'lr': LEARNING_RATE},
            {'params': feature_encoder.feature_layers.parameters(), },
            {'params': feature_encoder.fc1.parameters(), 'lr': LEARNING_RATE},
            {'params': feature_encoder.fc2.parameters(), 'lr': LEARNING_RATE},
            {'params': feature_encoder.head1.parameters(), 'lr': LEARNING_RATE},
            {'params': feature_encoder.head2.parameters(), 'lr': LEARNING_RATE},
            {'params': G_net.Net1.parameters(), 'lr': LEARNING_RATE},
            {'params': G_net.Net2.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE , momentum=momentum, weight_decay=l2_decay)
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
                use_spatial=True,
                spatial_window=3,
                min_spatial_agree=0.50,
                spatial_weight=0.20,
                target_rows=Row[RandPerm],
                target_cols=Column[RandPerm],
                target_height=G.shape[0],
                target_width=G.shape[1],
                use_multiview=True,
            )

            if tdus_info.get("spatial_status", "") in ("disabled_missing_coordinates", "disabled_invalid_coordinates"):
                print("[SC-DAPT-TDUS] spatial disabled due to missing coordinates:", tdus_info["spatial_status"])

            if tdus_info["num_selected"] == 0:
                labeled_loader = None
                print(
                    "[SC-DAPT-TDUS] Skip pseudo-label fine-tuning: "
                    f"reason={tdus_info['skip_reason']}, "
                    f"hist={tdus_info['class_hist']}, "
                    f"coverage={tdus_info['coverage']}, "
                    f"max_prior={tdus_info['max_prior']:.4f}, "
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

            print(
                "[SC-DAPT-TDUS][{}] epoch={}, selected={}, class_hist={}, coverage={}, pred_prior={}, max_prior={:.4f}, score_threshold={:.4f}, min_class_count={}, spa_agree={:.4f}, skip_reason={}, conf={:.4f}, ent={:.4f}, proto_dist={:.4f}, agree={:.4f}, score={:.4f}".format(
                    dataset_name,
                    epoch,
                    tdus_info["num_selected"],
                    tdus_info["class_hist"],
                    tdus_info["coverage"],
                    ["{:.3f}".format(x) for x in tdus_info["pred_prior"]],
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

        num_iter = len_source_loader

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

            #D_opt.zero_grad()
            D_opt.zero_grad()
            G_opt.zero_grad()
            source_aug_img1, source_aug_img2 = G_net(source_data00)
            G_opt.zero_grad()
            target_aug_img1, target_aug_img2 = G_net(target_data00)
            alpha1 = np.random.beta(0.6, 0.6)
            alpha2 = np.random.beta(0.6, 0.6)
            alpha3 = 1 - alpha1 - alpha2
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

            softmax_output_t = nn.Softmax(dim=1)(target_outputs).detach()
            _, pseudo_label_t = torch.max(softmax_output_t, 1)

            # Supervised Contrastive Loss
            all_source_con_features = torch.cat([source2.unsqueeze(1), source3.unsqueeze(1)], dim=1)  # 32  2 128
            # print("all_source_con_features",all_source_con_features.shape)
            all_target_con_features = torch.cat([target2.unsqueeze(1), target3.unsqueeze(1)], dim=1)  # 32 2 128
            # print("all_target_con_features",all_target_con_features.shape)
            lmmd_loss = mmd.lmmd(source_features, target_features, source_label,
                                 torch.nn.functional.softmax(target_outputs, dim=1), BATCH_SIZE=BATCH_SIZE,
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
            loss_correlation_alignment_loss = correlation_alignment_loss(source1, target1)

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

            loss = cls_loss + 0.01 * lambd * lmmd_loss + contrastive_loss_s + contrastive_loss_t + domain_similar_loss + loss_correlation_alignment_loss + loss_min
            # print(loss)

            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            # loss_min.backward()
            # NOTE: The original-like code updates feature_encoder with both D_opt and optimizer.
            # Keep this behavior unchanged for baseline comparability.
            D_opt.step()
            optimizer.step()
            teacher_encoder.update(feature_encoder)

            pred = source_outputs.data.max(1)[1]
            total_hit += pred.eq(source_label.data).sum()
            size += source_label.data.size()[0]
            test_accuracy = 100. * float(total_hit) / size
            if use_cbp_tdus:

                if i % len_update_loader == 0:
                    iter_update = iter(labeled_loader)
                clean_data, clean_label, clean_weight = next(iter_update)
                clean_data = clean_data.cuda().float()
                clean_label = clean_label.cuda().long()
                clean_weight = clean_weight.cuda().float()

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

                test_loss = 0.1 * loss_pseudo + 0.1 * loss_cons

                optimizer.zero_grad()
                test_loss.backward()
                optimizer.step()
                teacher_encoder.update(feature_encoder)

        print(
            'epoch {:>3d}:   cls loss: {:6.4f}, lmmd loss:{:6.4f}, occ loss:{:6f}, con_s loss:{:6f}, con_t loss:{:6f}, acc {:6.4f}, loss_min:{:6.4f}, cal_loss:{:6.4f}, total loss: {:6.4f}'
                .format(epoch, cls_loss.item(), lmmd_loss.item(), domain_similar_loss.item(), contrastive_loss_s.item(),
                        contrastive_loss_t.item(),
                        total_hit / size, loss_min.item(), loss_correlation_alignment_loss.item(), loss.item()))

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
