import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FEGC_Network(nn.Module):

    def __init__(self):
        super(FEGC_Network, self).__init__()

        self.conv1 = ConvBNReLU(in_channel=1, out_channel=32, kernel_size=3, stride=1, padding=1)

        self.cat_conv1 = ConvBNReLU(in_channel=32 + 64, out_channel=32, kernel_size=3, stride=1, padding=1)
        self.cat_conv2 = ConvBNReLU(in_channel=64 + 128, out_channel=64, kernel_size=3, stride=1, padding=1)
        self.cat_conv3 = ConvBNReLU(in_channel=128 + 526, out_channel=128, kernel_size=3, stride=1, padding=1)
        self.cat_conv4 = ConvBNReLU(in_channel=256 + 512, out_channel=256, kernel_size=3, stride=1, padding=1)
        self.cat_conv5 = ConvBNReLU(in_channel=512 + 512, out_channel=512, kernel_size=3, stride=1, padding=1)
        self.cat_conv6 = ConvBNReLU(in_channel=256 + 256, out_channel=256, kernel_size=3, stride=1, padding=1)
        self.cat_conv7 = ConvBNReLU(in_channel=128 + 128, out_channel=128, kernel_size=3, stride=1, padding=1)
        self.cat_conv8 = ConvBNReLU(in_channel=64 + 64, out_channel=64, kernel_size=3, stride=1, padding=1)

        self.block1 = ResBlock(in_channel=32, out_channel=64)
        self.block2 = ResBlock(in_channel=64, out_channel=128)
        self.block3 = ResBlock(in_channel=128, out_channel=256)
        self.block4 = ResBlock(in_channel=256, out_channel=512)

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.sgam = SGAM()

        self.dcm1 = DCM(de_channels=128, en_channels=64)
        self.dcm2 = DCM(de_channels=256, en_channels=128)
        self.dcm3 = DCM(de_channels=512, en_channels=256)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 32)
        self.fc2 = nn.Linear(32, 3)

        self.seg = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, pre_e1=None, pre_e2=None, pre_e3=None, pre_e4=None, pre_d4=None, pre_d3=None, pre_d2=None,
                pre_d1=None):

        if pre_e1 is not None:

            fegc_e1 = self.conv1(x)

            mix_e1 = torch.cat([fegc_e1, pre_e1], dim=1)

            mix_e1 = self.cat_conv1(mix_e1)

            mix_e1 = self.block1(mix_e1)

            fegc_e2 = self.pool(mix_e1)

            mix_e2 = torch.cat([fegc_e2, pre_e2], dim=1)

            mix_e2 = self.cat_conv2(mix_e2)

            mix_e2 = self.block2(mix_e2)

            fegc_e3 = self.pool(mix_e2)

            mix_e3 = torch.cat([fegc_e3, pre_e3], dim=1)

            mix_e3 = self.cat_conv3(mix_e3)

            mix_e3 = self.block3(mix_e3)

            fegc_e4 = self.pool(mix_e3)

            mix_e4 = torch.cat([fegc_e4, pre_e4], dim=1)

            mix_e4 = self.cat_conv4(mix_e4)

            mix_e4 = self.block4(mix_e4)

            fegc_d5 = self.sgam(mix_e4)

            images_level_prediction = self.avg_pool(fegc_d5)

            images_level_prediction = torch.flatten(images_level_prediction, 1)

            images_level_prediction = self.fc2(self.fc1(images_level_prediction))

            mix_d4 = torch.cat([fegc_d5, pre_d4], dim=1)

            fegc_d4 = self.cat_conv5(mix_d4)

            fegc_d4 = self.dcm3(mix_e3, fegc_d4)

            mix_d3 = torch.cat([fegc_d4, pre_d3], dim=1)

            fegc_d3 = self.cat_conv6(mix_d3)

            fegc_d3 = self.dcm2(mix_e2, fegc_d3)

            mix_d2 = torch.cat([fegc_d3, pre_d2], dim=1)

            fegc_d2 = self.cat_conv7(mix_d2)

            fegc_d2 = self.dcm1(mix_e1, fegc_d2)

            mix_d1 = torch.cat([fegc_d2, pre_d1], dim=1)

            fegc_d1 = self.cat_conv8(mix_d1)

            pixels_level_prediction = self.seg(fegc_d1)

            return pixels_level_prediction, images_level_prediction, fegc_d1
        else:

            fegc_e1 = self.conv1(x)

            fegc_e1 = self.block1(fegc_e1)

            fegc_e2 = self.pool(fegc_e1)

            fegc_e2 = self.block2(fegc_e2)

            fegc_e3 = self.pool(fegc_e2)

            fegc_e3 = self.block3(fegc_e3)

            fegc_e4 = self.pool(fegc_e3)

            fegc_e4 = self.block4(fegc_e4)

            fegc_d4 = self.sgam(fegc_e4)

            images_level_prediction = self.avg_pool(fegc_d4)

            images_level_prediction = torch.flatten(images_level_prediction, 1)

            images_level_prediction = self.fc2(self.fc1(images_level_prediction))

            fegc_d3 = self.dcm3(fegc_e3, fegc_d4)

            fegc_d2 = self.dcm2(fegc_e2, fegc_d3)

            fegc_d1 = self.dcm1(fegc_e1, fegc_d2)

            pixels_level_prediction = self.seg(fegc_d1)
            return pixels_level_prediction, images_level_prediction


class ConvBNReLU(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class SGAM(nn.Module):

    def __init__(self):
        super(SGAM, self).__init__()

    def forward(self, x):
        B, C, H, W = x.size()

        x = x.view(C, -1)
        x = torch.transpose(x, 0, 1)

        adj_matrix_A = torch.zeros(B * H * W, B * H * W)

        coords = torch.stack(torch.meshgrid(torch.arange(B), torch.arange(H), torch.arange(W)), dim=-1).view(B * H * W,
                                                                                                             3)

        dist = torch.abs(coords.unsqueeze(1) - coords.unsqueeze(0)).sum(dim=-1)

        row, col = torch.where(dist == 1)

        adj_matrix_A[row, col] = 1

        dig_matrix = torch.eye(B * H * W, B * H * W)
        adj_matrix_A = torch.add(adj_matrix_A, dig_matrix)

        adj_matrix_B = F.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=-1)

        adj_matrix_mean = torch.add(adj_matrix_A, adj_matrix_B) / 2
        adj_matrix_mean = F.softmax(adj_matrix_mean, dim=1)

        x_final = torch.einsum('ij,ii->ij', x, adj_matrix_mean)

        x_final = torch.relu(x_final)
        x_final = x_final.reshape(B, C, H, W)

        return x_final


class DCM(nn.Module):

    def __init__(self, de_channels, en_channels):
        super(DCM, self).__init__()

        self.conv1 = nn.Conv2d(en_channels, en_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(de_channels, en_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(en_channels, en_channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(de_channels * 2, en_channels, kernel_size=1)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)

    def forward(self, e, d):
        e1 = self.conv1(e)
        d1 = self.up(self.conv2(d))

        f = e1 + d1
        f = self.conv3(f)

        p_avg = self.avgpool(d)
        p_max = self.maxpool(d)

        p_all = torch.cat([p_avg, p_max], dim=1)

        p_all = self.conv4(p_all)

        out = p_all * f + d1

        return out


class ResBlock(nn.Module):

    def __init__(self, in_channel, out_channel, stride=1):
        super(ResBlock, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel)
        )

        self.shortcut = nn.Sequential()
        if in_channel != out_channel or stride > 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        output1 = self.layer(x)

        output2 = self.shortcut(x)

        output3 = output1 + output2

        output = F.relu(output3)
        return output
