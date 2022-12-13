import torch
import numpy as np

class iouEval:

    def __init__(self, nClasses, ignoreIndex=19):
        self.nClasses = nClasses
        self.ignoreIndex = ignoreIndex if nClasses>ignoreIndex else -1 #if ignoreIndex is larger than nClasses, consider no ignoreIndex
        self.reset()

    def reset (self):
        classes = self.nClasses if self.ignoreIndex==-1 else self.nClasses-1
        self.tp = torch.zeros(classes).double()
        self.fp = torch.zeros(classes).double()
        self.fn = torch.zeros(classes).double()        

    def addBatch(self, x, y):   #x=preds, y=targets
        # print("x.shape, y.shape", x.shape, y.shape, torch.unique(x), torch.unique(y))
        #sizes should be "batch_size x nClasses x H x W"
        
        #print ("X is cuda: ", x.is_cuda)
        #print ("Y is cuda: ", y.is_cuda)
        # print('testing nClasses inside iouEval.addBatch::::::: ', self.nClasses) 

        if (x.is_cuda or y.is_cuda):
            x = x.cuda()
            y = y.cuda()

        #if size is "batch_size x 1 x H x W" scatter to onehot
        if (x.size(1) == 1):
            x_onehot = torch.zeros(x.size(0), self.nClasses, x.size(2), x.size(3))  
            if x.is_cuda:
                x_onehot = x_onehot.cuda()
            x_onehot.scatter_(1, x, 1).float()
        else:
            x_onehot = x.float()

        if (y.size(1) == 1):
            y_onehot = torch.zeros(y.size(0), self.nClasses, y.size(2), y.size(3))
            if y.is_cuda:
                y_onehot = y_onehot.cuda()
            y_onehot.scatter_(1, y, 1).float()
        else:
            y_onehot = y.float()

        if (self.ignoreIndex != -1): 
            ignores = y_onehot[:,self.ignoreIndex].unsqueeze(1)
            x_onehot = x_onehot[:, :self.ignoreIndex]
            y_onehot = y_onehot[:, :self.ignoreIndex]
        else:
            ignores=0

        #print(type(x_onehot))
        #print(type(y_onehot))
        #print(x_onehot.size())
        #print(y_onehot.size())

        tpmult = x_onehot * y_onehot    #times prediction and gt coincide is 1
        tp = torch.sum(torch.sum(torch.sum(tpmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze()
        fpmult = x_onehot * (1-y_onehot-ignores) #times prediction says its that class and gt says its not (subtracting cases when its ignore label!)
        fp = torch.sum(torch.sum(torch.sum(fpmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze()
        fnmult = (1-x_onehot) * (y_onehot) #times prediction says its not that class and gt says it is
        fn = torch.sum(torch.sum(torch.sum(fnmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze()
        # tnmult = (1-x_onehot) * (1 - y_onehot - ignores)
        # tn = torch.sum(torch.sum(torch.sum(tnmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze()

        self.tp += tp.double().cpu()
        self.fp += fp.double().cpu()
        self.fn += fn.double().cpu()
        # self.tn += tn.double().cpu()

    def getIoU(self):
        num = self.tp
        den = self.tp + self.fp + self.fn + 1e-15
        iou = num / den
        print('check:::::::::size of io tensor: ', iou.size())
        return torch.mean(iou), iou     #returns "iou mean", "iou per class"

    def getPre(self):
        num = self.tp
        den = self.tp + self.fp + 1e-15
        pre = num / den
        print('check:::::::::size of io tensor: ', pre.size())
        return pre

    def getRecall(self):
        num = self.tp
        den = self.tp + self.fn + 1e-15
        recall = num / den
        print('check:::::::::size of io tensor: ', recall.size())
        return recall

    def getF1(self):
        pre = self.tp / (self.tp + self.fp + 1e-15)
        recall = self.tp / (self.tp + self.fn + 1e-15)
        f1 = (2 * pre * recall) / pre + recall
        print('check:::::::::size of io tensor: ', f1.size())
        return f1

# Class for colors
class colors:
    RED       = '\033[31;1m'
    GREEN     = '\033[32;1m'
    YELLOW    = '\033[33;1m'
    BLUE      = '\033[34;1m'
    MAGENTA   = '\033[35;1m'
    CYAN      = '\033[36;1m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC      = '\033[0m'

# Colored value output if colorized flag is activated.
def getColorEntry(val):
    if not isinstance(val, float):
        return colors.ENDC
    if (val < .20):
        return colors.RED
    elif (val < .40):
        return colors.YELLOW
    elif (val < .60):
        return colors.BLUE
    elif (val < .80):
        return colors.CYAN
    else:
        return colors.GREEN


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
