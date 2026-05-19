import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from sklearn import metrics
from TopoAlignLoss import ImprovedTopoAlignLoss
import time
import utils
from torch.utils.data import TensorDataset, DataLoader
from config_SH2HZ import *
from net import F_encoder
from TAMVCPO import MVCPO_Helper,TopologyValidator
import warnings
warnings.filterwarnings('ignore')
##################################
file_path = './datasets/Shanghai-Hangzhou/DataCube.mat'
data_s, data_t, label_s, label_t = utils.cubeData(file_path)
# Loss Function
loss_function = nn.MSELoss().cuda()
crossEntropy = nn.CrossEntropyLoss().cuda()
classification_loss = nn.CrossEntropyLoss().cuda()
# 实例化熵损失函数
entropy_loss = utils.EntropyLoss()
acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, CLASS_NUM])
k = np.zeros([nDataSet, 1])
best_predict_all = []
best_acc_all = 0.0
best_G,best_RandPerm,best_Row, best_Column,best_nTrain = None,None,None,None,None

for iDataSet in range(nDataSet):
    print('#######################idataset######################## ', iDataSet)

    utils.seed_everything(seeds[iDataSet], use_deterministic=True)

    trainX, trainY = utils.get_sample_data(data_s, label_s, HalfWidth, 180)
    testID, testX, testY, G, RandPerm, Row, Column = utils.get_all_data(data_t, label_t, HalfWidth)

    train_dataset = TensorDataset(torch.tensor(trainX), torch.tensor(trainY))
    test_dataset = TensorDataset(torch.tensor(testX), torch.tensor(testY))

    train_loader_s = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    train_loader_t = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    len_source_loader = len(train_loader_s)
    len_target_loader = len(train_loader_t)
    topo_validator = TopologyValidator(CLASS_NUM)
    mvpo_helper = MVCPO_Helper(CLASS_NUM)
    loss_fn = ImprovedTopoAlignLoss(base_k=2, max_k=16, temp_base=0.4, feat_dim=128, memory_size=32).cuda()
    feature_encoder = F_encoder(BATCH_SIZE,nBand, patch_size, CLASS_NUM,num_tokens=1).cuda()
    print("Training...")
    last_accuracy = 0.0
    best_episdoe = 0
    train_loss = []
    test_acc = []
    running_D_loss, running_F_loss = 0.0, 0.0
    running_label_loss = 0
    running_domain_loss = 0
    total_hit, total_num = 0.0, 0.0
    size = 0.0
    test_acc_list = []
    total_hit_t,total_num_t=0.0,0.0
    cls_losses = []
    total_losses = []
    ACC = []
    epoch_weights = []
    train_start = time.time()
    # loss plot
    loss1 = []
    loss2 = []
    loss3 = []
    for epoch in range(1, epochs + 1):

        LEARNING_RATE = lr  # / math.pow((1 + 10 * (epoch - 1) / epochs), 0.75)
        print('learning rate{: .4f}'.format(LEARNING_RATE))
        optimizer = torch.optim.SGD([
            {'params': feature_encoder.multi_level_feature_selector.parameters(), },
            {'params': feature_encoder.fc2.parameters(), 'lr': LEARNING_RATE},
            {'params': feature_encoder.head1.parameters(), 'lr': LEARNING_RATE},
            {'params': feature_encoder.head2.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE, momentum=momentum, weight_decay=l2_decay)

        cls_criterion = torch.nn.CrossEntropyLoss()
        feature_encoder.train()
        loss_fn.update_epoch(epoch)  # 更新epoch计数

        iter_source = iter(train_loader_s)
        iter_target = iter(train_loader_t)

        num_iter = len_source_loader
        for i in range(1, num_iter):

            source_data, source_label = next(iter_source)
            target_data, _ = next(iter_target)
            if i % len_target_loader == 0:
                iter_target = iter(train_loader_t)

            source, source1, source_output_classes, source2, source3, low_retention_loss_s, \
            target, target1, target_output_classes, target2, target3, high_retention_loss_t \
                = feature_encoder(source_data.cuda(), target_data.cuda())

            source_data1 = utils.flip_augmentation(source_data)
            target_data1 = utils.flip_augmentation(target_data)

            _, s1, _,  _, _, sl1, _, t3, t1, _,  _, tl1, = feature_encoder(source_data1.cuda(), target_data.cuda())

            _, s2, _,  _, _, sl2, _, t4, t2, _, _, tl2, = feature_encoder(source_data.cuda(), target_data1.cuda())
            softmax_output_t = nn.Softmax(dim=1)(target_output_classes).detach()
            _, pseudo_label_t = torch.max(softmax_output_t, 1)
            pseudo_labels, selected_mask, _ = mvpo_helper.select_pseudo_labels(
                target_output_classes, t1, t2, epoch
            )
            if epoch % 10 == 0 and len(pseudo_labels) == BATCH_SIZE:
                invalid_mask, avg_topo_diff = topo_validator.validate_pseudo_labels(
                    source_output_classes.detach(),
                    source_label.cuda(),
                    target_output_classes.detach(),
                    pseudo_labels, epoch)

                pseudo_labels, selected_mask, _ = mvpo_helper.select_pseudo_labels(target_output_classes, t1, t2, epoch, avg_topo_diff )
            if len(pseudo_labels) > 0:
                target_cls_loss = crossEntropy(
                    target_output_classes[selected_mask],
                    pseudo_labels.cuda()
                )
            else:
                target_cls_loss = torch.tensor(0.0).cuda()
            # === 损失计算 ===
            loss_s = crossEntropy(source_output_classes, source_label.cuda())
            loss_fn.update_memory(source1.detach(), target1.detach())
            loss_tal = loss_fn(source1, target1)
            #总损失
            loss = loss_s + low_retention_loss_s + high_retention_loss_t + loss_tal + target_cls_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = source_output_classes.data.max(1)[1]
            total_hit += pred.eq(source_label.data.cuda()).sum()
            size += source_label.data.size()[0]
            test_accuracy = 100. * float(total_hit) / size

        print(
            'epoch {:>3d}:  loss_cls: {:6.4f}, acc: {:6.4f}, total loss: {:6.4f}'
                .format(epoch, loss_s.item(),
                        total_hit / size, loss.item()))

        total_losses.append(loss.item())
        ACC.append((total_hit / size).item())
        train_end = time.time()

        if epoch % epochs == 0:
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
                    _,_,test_outputs,_ = feature_encoder(Variable(source_data).cuda(),Variable(test_datas).cuda(),false='test')
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
            acc[iDataSet] = 100. * total_rewards / len(test_loader.dataset)
            OA = acc
            C = metrics.confusion_matrix(labels, predict)
            A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float64)

            k[iDataSet] = metrics.cohen_kappa_score(labels, predict)
            print('\t\tAccuracy: {}/{} ({:.2f}%)\n'.format(total_rewards, len(test_loader.dataset),
                                                           100. * total_rewards / len(test_loader.dataset)))
            test_end = time.time()
            # Training mode
            if test_accuracy > last_accuracy:
                # save networks
                print("save networks for epoch:", epoch + 1)
                last_accuracy = test_accuracy
                best_episdoe = epoch
                best_predict_all = predict
                best_G, best_RandPerm, best_Row, best_Column = G, RandPerm, Row, Column
                print('best epoch:[{}], best accuracy={}'.format(best_episdoe + 1, last_accuracy))

            print('iter:{} best epoch:[{}], best accuracy={}'.format(iDataSet, best_episdoe + 1, last_accuracy))
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
print ("average OA: " + "{:.2f}".format( OAMean) + " +- " + "{:.2f}".format( OAStd))
print ("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
print ("average kappa: " + "{:.4f}".format(100 *kMean) + " +- " + "{:.4f}".format(100 *kStd))
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

