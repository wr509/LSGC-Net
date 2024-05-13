import torch

import torch.nn as nn

from torch.nn import functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LAMLoss(nn.Module):
    def __init__(self, in_channel=4, out_channel=256, num=1500, pos_num=128, neg_num=256, m=0.5, tau=0.1):
        super(LAMLoss, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num = num

        self.pos_num = pos_num
        self.neg_num = neg_num

        self.m = m

        self.tau = tau
        self.fc = nn.Linear(self.in_channel, self.out_channel)

    def forward(self, d1_fegc, p_fegc, p_pre, gt):
        b, c, h, w = d1_fegc.size()

        p_overlap = torch.zeros(b, 6, h, w)

        one_hot_fegc = torch.where(p_fegc >= 0.5, 1, 0)

        max_values, max_indices = torch.max(p_pre, dim=1)

        one_hot_pre = torch.zeros_like(p_pre)

        one_hot_pre.scatter_(1, max_indices.unsqueeze(1), 1)

        p_overlap[:, 0, :, :] = torch.where((one_hot_fegc[:, 0, :, :] == 0) & (one_hot_pre[:, 0, :, :] == 1), 1, 0)

        p_overlap[:, 1, :, :] = torch.where((one_hot_fegc[:, 0, :, :] == 1) & (one_hot_pre[:, 1, :, :] == 1), 1, 0)

        p_overlap[:, 2, :, :] = torch.where((one_hot_fegc[:, 0, :, :] == 0) & (one_hot_pre[:, 2, :, :] == 1), 1, 0)

        p_overlap[:, 3, :, :] = torch.where((one_hot_fegc[:, 0, :, :] == 1) & (one_hot_pre[:, 0, :, :] == 1), 1, 0)

        p_overlap[:, 4, :, :] = torch.where((one_hot_fegc[:, 0, :, :] == 0) & (one_hot_pre[:, 1, :, :] == 1), 1, 0)

        p_overlap[:, 5, :, :] = torch.where((one_hot_fegc[:, 0, :, :] == 1) & (one_hot_pre[:, 2, :, :] == 1), 1, 0)

        pre_region = (p_overlap[:, 3, :, :] != 0).float().unsqueeze(1).bool()

        pre_point_indices = torch.nonzero(pre_region)

        pre_point = torch.masked_select(d1_fegc, pre_region)

        pre_point = torch.transpose(pre_point, 0, 1)

        if pre_point_indices.size(0) < self.num * b:

            edge_extract = EdgeExtract()

            edge_region = edge_extract(gt).bool()

            edge_point_indices = torch.nonzero(edge_region)

            edge_point = torch.masked_select(d1_fegc, edge_region)

            edge_point = edge_point.view(c, -1)

            edge_point = torch.transpose(edge_point, 0, 1)

            all_point = torch.cat((pre_point, edge_point), dim=0)

            all_point_indices = torch.cat((pre_point_indices, edge_point_indices), dim=0)


        else:
            all_point = pre_point
            all_point_indices = pre_point_indices

        gt = gt.bool()

        gt_point_indices = torch.nonzero(gt)

        eq = torch.eq(all_point_indices.unsqueeze(1), gt_point_indices)

        all = torch.all(eq, dim=-1)

        mask = torch.where(all)[0]

        pos_point_indices = all_point_indices[mask]

        isin = torch.isin(all_point_indices, pos_point_indices)
        not_isin = torch.logical_not(isin)
        neg_point_indices = torch.masked_select(all_point_indices, not_isin).view(-1, 4)

        all_eq_pos = torch.eq(all_point_indices.unsqueeze(1), pos_point_indices.unsqueeze(0))

        all_eq_pos_all = torch.all(all_eq_pos, dim=-1)

        all_point_indices_mask = torch.any(all_eq_pos_all, dim=1)

        pos_point_index = torch.nonzero(all_point_indices_mask, as_tuple=False).squeeze()
        all_eq_neg = torch.eq(all_point_indices.unsqueeze(1), neg_point_indices.unsqueeze(0))

        all_eq_neg_all = torch.all(all_eq_neg, dim=-1)

        all_point_indices_neg_mask = torch.any(all_eq_neg_all, dim=1)

        neg_point_index = torch.nonzero(all_point_indices_neg_mask, as_tuple=False).squeeze()
        pos_point = torch.index_select(all_point, dim=0, index=pos_point_index)
        neg_point = torch.index_select(all_point, dim=0, index=neg_point_index)
        pos_point = self.fc(pos_point)
        neg_point = self.fc(neg_point)

        pos_point = torch.index_select(pos_point, 0, torch.randperm(pos_point.size(0))[:self.pos_num * b])

        neg_point = torch.index_select(neg_point, 0, torch.randperm(neg_point.size(0))[:self.neg_num * b])

        point = torch.cat((pos_point, neg_point), dim=0)

        cos_sim = F.cosine_similarity(point.unsqueeze(1), point.unsqueeze(0), dim=-1)

        mask_self = torch.eye(self.pos_num * b + self.pos_num * b).bool()

        pos_neg_label = torch.ones(self.pos_num * b + self.pos_num * b, self.pos_num * b + self.pos_num * b)
        pos_neg_label[:self.pos_num * b, self.pos_num * b:] = 0
        pos_neg_label[self.pos_num * b:, :self.pos_num * b] = 0

        lam_loss = (1 - pos_neg_label) * (cos_sim / self.tau).pow(2) + pos_neg_label * torch.clamp(
            self.m - cos_sim / self.tau, min=0).pow(2)
        lam_loss = lam_loss.masked_fill(mask_self, 0)

        lam_loss = lam_loss.sum() / (lam_loss != 0).sum()

        return lam_loss


class EdgeExtract(nn.Module):
    def __init__(self):
        super(EdgeExtract, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = x.float()
        pooled_x = self.pool(x)
        edge = torch.abs(x - pooled_x)
        edge_bool = edge.bool()
        edge = edge.masked_fill(~edge_bool, 0)
        edge = edge.long()
        return edge

    d1_fegc = torch.rand(1, 4, 6, 6)
    print("d1_fegc=", d1_fegc)
    p_fegc = torch.rand(1, 1, 6, 6)
    print("p_fegc=", p_fegc)
    p_pre = torch.rand(1, 3, 6, 6)
    print("p_pre=", p_pre)
    gt = torch.zeros(1, 1, 6, 6)
    gt[:, :, 2:4, 2:4] = 1

    loss = LAMLoss()
    y = loss(d1_fegc, p_fegc, p_pre, gt)
