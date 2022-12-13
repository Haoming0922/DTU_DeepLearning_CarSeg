import torch.nn.functional as F
import torch
import torch.nn as nn
from utils.utils import FocalLoss, dice_loss
'''Feature map loss
'''

def fm(x):
    out = F.normalize(x.pow(2).mean(1).view(x.size(0), -1))
    return out

def fm_loss(x, y):
    return (fm(x) - fm(y)).pow(2).mean()


def similarity_loss(f_s, f_t):
    bsz = f_s.shape[0]
    f_s = f_s.view(bsz, -1)
    f_t = f_t.view(bsz, -1)

    # print('f_s.shape, f_t.shape: ', f_s.shape, f_t.shape)
    G_s = torch.mm(f_s, torch.t(f_s))
    # print('G_s: ', G_s)
    # G_s = G_s / G_s.norm(2)
    G_s = torch.nn.functional.normalize(G_s)
    G_t = torch.mm(f_t, torch.t(f_t))
    # print('G_t: ', G_t)
    # G_t = G_t / G_t.norm(2)
    G_t = torch.nn.functional.normalize(G_t)

    G_diff = G_t - G_s
    # print('G_t, G_s, G_diff: ', G_t, G_s, G_diff)
    loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
    return loss

def cos_similarity_loss(f_s, f_t):
    bsz = f_s.shape[0]
    f_s = f_s.view(bsz, -1)
    f_t = f_t.view(bsz, -1)

    # cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    #
    # cos_loss = cos(f_s, f_t)
    # cos_loss = sum(cos_loss) / len(cos_loss)

    cos_simi = torch.cosine_similarity(f_s, f_t, dim=0)
    cos_simi = sum(cos_simi) / len(cos_simi)
    cos_loss = 1 - cos_simi
    return cos_loss

def hybrid_loss(predictions, target):
    """Calculating the loss"""
    loss = 0

    # gamma=0, alpha=None --> CE
    focal = FocalLoss(gamma=0, alpha=None)

    for prediction in predictions:

        bce = focal(prediction, target)
        dice = dice_loss(prediction, target)
        loss += bce + dice

    return loss

'''from /home/a409/users/wenglean/my_progs/CIRKD/losses/skd.py
'''
class CriterionStructuralKD(nn.Module):
    def __init__(self):
        super(CriterionStructuralKD, self).__init__()

    def pair_wise_sim_map(self, fea):
        B, C, H, W = fea.size()
        fea = fea.reshape(B, C, -1)
        fea_T = fea.transpose(1, 2)
        sim_map = torch.bmm(fea_T, fea)
        return sim_map

    def forward(self, feat_S, feat_T):
        B, C, H, W = feat_S.size()

        # feat_S = feat_S.reshape(B, C, -1)
        patch_w = 2  # int(0.5 * W)
        patch_h = 2  # int(0.5 * H)
        maxpool = nn.MaxPool2d(kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w), padding=0, ceil_mode=True)
        feat_S = maxpool(feat_S)
        feat_T = maxpool(feat_T)

        feat_S = F.normalize(feat_S, p=2, dim=1)
        feat_T = F.normalize(feat_T, p=2, dim=1)

        # S_sim_map = self.pair_wise_sim_map(feat_S)
        # T_sim_map = self.pair_wise_sim_map(feat_T)
        S_sim_map = feat_S
        T_sim_map = feat_T
        # B, H, W = S_sim_map.size()

        sim_err = ((S_sim_map - T_sim_map) ** 2)
        sim_dis = sim_err.mean()

        return sim_dis

class CrossEntropyLoss2d(torch.nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss = torch.nn.NLLLoss2d(weight)

    def forward(self, outputs, targets):
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)


if __name__ == "__main__":
    a_1 = torch.tensor([[[1., 2.], [3., 4.]],
                        [[3., 1.], [5., 2.]]])
    a_2 = torch.tensor([[[1., 3.], [3., 6.]],
                        [[1., 3.], [3., 6.]]])

    b_1 = torch.tensor([[[10., 11.], [6., 7.]],
                        [[7., 6.], [2., 4.]]])
    b_2 = torch.tensor([[[9., 8.], [4., 6.]],
                        [[9., 10.], [4., 5.]]])


    fm1 = [a_1, a_2]
    fm2 = [b_1, b_2]
    losses_group = [fm_loss(x, y) for x, y in zip(fm1, fm2)]
    print(losses_group)
    print(sum(losses_group))

    ab_1 = torch.cat((a_1, b_1), dim=0)
    ab_2 = torch.cat((a_2, b_2), dim=0)
    print('ab_2.shape', ab_2.shape)
    sum_cat = fm_loss(ab_1, ab_2)
    print('sum_cat: ', sum_cat)

    ab_de1 = a_1 - b_1
    ab_de2 = a_2 - b_2
    sum_dec = fm_loss(ab_de1, ab_de2)
    print('sum_dec: ', sum_dec)

    ab_plus1 = a_1 + b_1
    ab_plus2 = a_2 + b_2
    sum_plus = fm_loss(ab_plus1, ab_plus2)
    print('sum_plus: ', sum_plus)

    print(ab_plus1.shape, ab_plus2.shape)
    simi_loss = similarity_loss(ab_plus1, ab_plus2)
    print('simi_loss: ', simi_loss)

    cross_simi = similarity_loss(a_1, b_1) + similarity_loss(a_2, b_2)
    print('cross_simi:', cross_simi)


    cos_loss = cos_similarity_loss(a_1, b_1) + cos_similarity_loss(a_2, b_2)
    print('cos_loss: ', cos_loss)



