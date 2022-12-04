import numpy as np
import math
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import os
import cv2

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def get_hist(image, label, num_class):
    hist = np.zeros((num_class, num_class))
    hist += fast_hist(image.flatten(), label.flatten(), num_class)
    return hist


def cal_kappa(hist):
    if hist.sum() == 0:
        po = 0
        pe = 1
        kappa = 0
    else:
        po = np.diag(hist).sum() / hist.sum()
        pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
        if pe == 1:
            kappa = 0
        else:
            kappa = (po - pe) / (1 - pe)
    return kappa



def SCDD_eval_all(preds, labels, num_class):
    hist = np.zeros((num_class, num_class))
    for pred, label in zip(preds, labels):
        infer_array = np.array(pred)
        unique_set = set(np.unique(infer_array))
        assert unique_set.issubset(set([0, 1, 2, 3, 4, 5, 6])), "unrecognized label number"
        label_array = np.array(label)
        # print('infer_array.shape:', infer_array.shape, 'label_array.shape:', label_array.shape)
        # print('infer_array.unique:', np.unique(infer_array), 'label_array.unique:', np.unique(label_array))
        assert infer_array.shape == label_array.shape, "The size of prediction and target must be the same"
        hist += get_hist(infer_array, label_array, num_class)

    hist_fg = hist[1:, 1:]
    c2hist = np.zeros((2, 2))
    c2hist[0][0] = hist[0][0]
    c2hist[0][1] = hist.sum(1)[0] - hist[0][0]
    c2hist[1][0] = hist.sum(0)[0] - hist[0][0]
    c2hist[1][1] = hist_fg.sum()
    hist_n0 = hist.copy()
    hist_n0[0][0] = 0
    kappa_n0 = cal_kappa(hist_n0)
    iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist))
    IoU_fg = iu[1]
    IoU_mean = (iu[0] + iu[1]) / 2
    Sek = (kappa_n0 * math.exp(IoU_fg)) / math.e
    Score = 0.3 * IoU_mean + 0.7 * Sek
    return Score, IoU_mean, Sek

def accuracy(pred, label, ignore_zero=False):
    valid = (label >= 0)
    if ignore_zero: valid = (label > 0)
    acc_sum = (valid * (pred == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, count, weight):
        self.val = val
        self.avg = val
        self.count = count
        self.sum = val * weight
        self.initialized = True

    def update(self, val, count=1, weight=1):
        if not self.initialized:
            self.initialize(val, count, weight)
        else:
            self.add(val, count, weight)

    def add(self, val, count, weight):
        self.val = val
        self.count += count
        self.sum += val * weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def create_visual_output(output):
    label2color_dict = {
        0: [255, 255, 255],
        1: [128,128,128],
        2: [0,128,0],
        3: [0,255,0],
        4: [128,0,0],
        5: [255,0,0],
        6: [0, 0, 255],
    }
    # visualize
    # print('output.shape', output.shape)
    visual_map = np.zeros((output.shape[0], output.shape[1], 3), dtype=np.uint8)
    for i in range(visual_map.shape[0]):
        for j in range(visual_map.shape[1]):
            color = label2color_dict[output[i, j]]
            visual_map[i, j, 0] = color[0]
            visual_map[i, j, 1] = color[1]
            visual_map[i, j, 2] = color[2]

    return visual_map


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        # self.confusion_matrix = np.zeros((self.num_class,)*2)
        self.confusion_matrix = torch.zeros(self.num_class, self.num_class)

    def Pixel_Accuracy(self):
        # Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        Acc = torch.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        # Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = torch.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum(dim=0).data.cpu().numpy()
        Acc = np.nanmean(Acc)
        # Acc = Acc.mean()
        return Acc

    def Mean_Intersection_over_Union(self):
        # MIoU = np.diag(self.confusion_matrix) / (
        #             np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
        #             np.diag(self.confusion_matrix))
        MIoU = torch.diag(self.confusion_matrix) / (
            self.confusion_matrix.sum(dim=1) + self.confusion_matrix.sum(dim=0) - torch.diag(self.confusion_matrix)
        ).data.cpu().numpy()
        MIoU = np.nanmean(MIoU)
        # print('MIoU: ', MIoU)
        # MIoU = MIoU.mean()
        return MIoU

    def Precision(self):
        Pre = self.confusion_matrix[1][1] / (self.confusion_matrix[0][1] + self.confusion_matrix[1][1]).data.cpu().numpy()
        return Pre

    def Recall(self):
        Re = self.confusion_matrix[1][1] / (self.confusion_matrix[1][1] + self.confusion_matrix[1][0]).data.cpu().numpy()
        return Re

    def F1(self):
        Pre = self.confusion_matrix[1][1] / (self.confusion_matrix[0][1] + self.confusion_matrix[1][1])
        Re = self.confusion_matrix[1][1] / (self.confusion_matrix[1][1] + self.confusion_matrix[1][0])
        F1 = 2 * Pre * Re / (Pre+Re)
        return F1

    def Frequency_Weighted_Intersection_over_Union(self):
        # freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        freq = self.confusion_matrix.sum(dim=1) / self.confusion_matrix.sum()
        # iu = np.diag(self.confusion_matrix) / (
        #             np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
        #             np.diag(self.confusion_matrix))
        iu = torch.diag(self.confusion_matrix) / (
            self.confusion_matrix.sum(dim=1) + self.confusion_matrix.sum(dim=0) -
            torch.diag(self.confusion_matrix))

        # FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU


    def generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].type(torch.IntTensor).cuda() + pre_image[mask]
        tn = (label == 0).type(torch.IntTensor).sum()
        fp = (label == 1).type(torch.IntTensor).sum()
        fn = (label == 2).type(torch.IntTensor).sum()
        tp = (label == 3).type(torch.IntTensor).sum()
        # print('tn, fp, fn, tp', tn, fp, fn, tp)
        confusion_matrix = torch.tensor([[tn, fp], [fn, tp]])
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        # assert gt_image.shape == pre_image.shape

        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self.generate_matrix(gt_image, pre_image)

    def reset(self):
        # self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.confusion_matrix = torch.zeros(self.num_class, self.num_class)



class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)

            # N,C,H*W => N,H*W,C
            input = input.transpose(1, 2)

            # N,H*W,C => N*H*W,C
            input = input.contiguous().view(-1, input.size(2))


        target = target.view(-1, 1)
        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

def dice_loss(logits, true, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)



'''
Visualize the feature map
'''

def various_distance(out_vec_t0, out_vec_t1,dist_flag):
    if dist_flag == 'l2':
        distance = F.pairwise_distance(out_vec_t0, out_vec_t1, p=2)
    if dist_flag == 'l1':
        distance = F.pairwise_distance(out_vec_t0, out_vec_t1, p=1)
    if dist_flag == 'cos':
        distance = 1 - F.cosine_similarity(out_vec_t0, out_vec_t1)
    return distance

def check_dir(dir):
    if not os.path.exists(dir):
        #os.mkdir(dir)
        os.makedirs(dir)

def single_layer_similar_heatmap_visual(output_t0,output_t1,save_change_map_dir,epoch,filename,layer_flag,dist_flag):
    interp = nn.Upsample(size=[256, 256], mode='bilinear')
    n, c, h, w = output_t0.data.shape

    print("output_t0.data.shape:", output_t0.data.shape)
    # out_t0 = out_t0.permute(0,2,3,1)
    out_t0_, out_t1_ = output_t0.squeeze(0), output_t1.squeeze(0)

    out_t0_rz = torch.transpose(out_t0_.view(c, h * w), 1, 0)
    out_t1_rz = torch.transpose(out_t1_.view(c, h * w), 1, 0)

    distance = various_distance(out_t0_rz,out_t1_rz,dist_flag=dist_flag)
    print(type(distance), distance.shape)
    similar_distance_map = distance.view(h,w).data.cpu().numpy()
    print("similar_distance_map.shape", similar_distance_map.shape, np.unique(similar_distance_map))
    similar_distance_map_rz = interp(Variable(torch.from_numpy(similar_distance_map[np.newaxis, np.newaxis, :])))
    print(torch.unique(similar_distance_map_rz))
    similar_dis_map_colorize = cv2.applyColorMap(np.uint8(255 * similar_distance_map_rz.data.cpu().numpy()[0][0]), cv2.COLORMAP_JET)
    save_change_map_dir_ = os.path.join(save_change_map_dir, 'epoch_' + str(epoch))
    check_dir(save_change_map_dir_)

    #print("save_change_map_dir_: ", save_change_map_dir_)
    save_change_map_dir_layer = os.path.join(save_change_map_dir_,layer_flag)

    save_weight_fig_dir = os.path.join(save_change_map_dir_layer, filename)
    save_weight_fig_dir_1 = os.path.dirname(save_weight_fig_dir)
    check_dir(save_weight_fig_dir_1)

    cv2.imwrite(save_weight_fig_dir, similar_dis_map_colorize)
    print("save_weight_fig_dir:", save_weight_fig_dir)
    cv2.imwrite(save_weight_fig_dir.replace('.jpg', '_gray.jpg'), np.uint8(255 * similar_distance_map_rz.data.cpu().numpy()[0][0]))

    return np.uint8(255 * similar_distance_map_rz.data.cpu().numpy())