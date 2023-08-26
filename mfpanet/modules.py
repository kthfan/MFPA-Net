import torch
import torch.nn as nn
import torch.nn.functional as F

from .drn import drn_a_50

__all__ = ['MFPN', 'DAM', 'ASPP', 'MFPANet']


class MFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=48):
        super().__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels

        self.conv_list = nn.ModuleList()
        for in_channels in self.in_channels_list:
            self.conv_list.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ))

    def forward(self, x_list):
        x0 = self.conv_list[-1](x_list[-1])
        out_list = [x0]
        for conv, x in zip(self.conv_list[-2::-1], x_list[-2::-1]):
            x = conv(x)
            if any([a != b for a, b in zip(x0.shape[2:], x.shape[2:])]):
                x0 = F.interpolate(x0, x.shape[2:])
            x0 = x0 + x
            out_list.append(x0)

        out_shape = out_list[-1].shape[2:]
        out_list = [F.interpolate(out, out_shape) if any([a != b for a, b in zip(out.shape[2:], out_shape)]) else out
                    for out in out_list]
        return torch.cat(out_list, dim=1)


class SAM(nn.Module):
    def __init__(self, in_channels, latent_channels=None):
        super().__init__()
        if latent_channels is None:
            latent_channels = in_channels // 8
        self.in_channels = in_channels
        self.latent_channels = latent_channels

        self.conv_qk = nn.Conv2d(in_channels, 2 * latent_channels, 1, bias=False)
        self.conv_v = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, _, *spatial = x.shape
        qk = self.conv_qk(x)
        qk = qk.view(b, 2 * self.latent_channels, -1)
        q, k = qk.split(self.latent_channels, dim=1)
        v = self.conv_v(x)
        v = v.view(b, self.in_channels, -1)

        scale = 1 / (self.latent_channels) ** 0.25
        m = torch.einsum("bcs, bct -> bst", scale * q, scale * k)
        m = F.softmax(m, dim=-1)
        a = torch.einsum('bst, bct -> bcs', m, v)
        a = a.view(b, self.in_channels, *spatial)
        return self.gamma * a + x


class CAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, ch, *spatial = x.shape
        y = x.view(b, ch, -1)

        scale = 1 / (y.shape[-1]) ** 0.25
        m = torch.einsum("bcs, bds -> bcd", scale * y, scale * y)
        m = F.softmax(m, dim=-1)
        a = torch.einsum('bcd, bds -> bcs', m, y)
        a = a.view(b, self.in_channels, *spatial)
        return self.gamma * a + x


class DAM(nn.Module):
    def __init__(self, in_channels, out_channels=None, sam_channels=None, cam_channels=None,
                 latent_channels=None, drop=0.1):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        if sam_channels is None:
            sam_channels = sam_channels
        if cam_channels is None:
            cam_channels = cam_channels
        if latent_channels is None:
            latent_channels = in_channels // 8
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sam_channels = sam_channels
        self.cam_channels = cam_channels
        self.latent_channels = latent_channels

        self.conv_sam = nn.Sequential(
            nn.Conv2d(in_channels, sam_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(sam_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_cam = nn.Sequential(
            nn.Conv2d(in_channels, cam_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(cam_channels),
            nn.ReLU(inplace=True),
        )
        self.sam = SAM(sam_channels, latent_channels)
        self.cam = CAM(cam_channels)
        self.conv_out = nn.Sequential(
            nn.Dropout2d(drop),
            nn.Conv2d(sam_channels + cam_channels, out_channels, 3, 1, 1, bias=False),
        )

    def forward(self, x):
        xs = self.conv_sam(x)
        xs = self.sam(xs)
        xc = self.conv_cam(x)
        xc = self.cam(xc)
        x = torch.cat([xs, xc], dim=1)
        x = self.conv_out(x)
        return x


#################################################################
# extracted from https://github.com/VainF/DeepLabV3Plus-Pytorch #
#################################################################
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, atrous_rates=[12, 24, 36]):
        super(ASPP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.atrous_rates = atrous_rates
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1), )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)



class MFPANet(nn.Module):
    def __init__(self, num_classes, pretrained=False, backbone_name='drn_a_50'):
        super().__init__()
        assert backbone_name in ['drn_a_50', 'drn_c_26','drn_c_42', 'drn_c_58', 'drn_d_24', 'drn_d_38', 'drn_d_40', 'drn_d_54', 'drn_d_56', 'drn_d_105', 'drn_d_107']
        self.num_classes = num_classes
        self.backbone_name = backbone_name

        if backbone_name == 'drn_a_50':
            self.mid_channels = [256, 512, 1024, 2048]
            self.latent_channels = 2048
            self.fpn_index = [1, 2, 3, 4]
        elif backbone_name in ['drn_c_26', 'drn_c_42', 'drn_d_24', 'drn_d_38', 'drn_d_40']:
            self.mid_channels = [64, 128, 256, 512]
            self.latent_channels = 512
            self.fpn_index = [2, 3, 4, 5]
        else:
            self.mid_channels = [256, 512, 1024, 2048]
            self.latent_channels = 512
            self.fpn_index = [2, 3, 4, 5]

        # set backbone
        self.backbone = getattr(drn, backbone_name)(pretrained=pretrained, num_classes=1000, out_middle=True)
        self.backbone.num_classes = None
        del self.backbone.avgpool
        del self.backbone.fc

        self.dam = DAM(self.latent_channels, out_channels = self.latent_channels,
                       sam_channels = self.latent_channels // 4, cam_channels = self.latent_channels // 4,
                       latent_channels = self.latent_channels // 4 // 8)
        self.aspp = ASPP(self.dam.out_channels, 256, [12, 24, 36])
        self.mfpn = MFPN(self.mid_channels, 48)
        self.classifier = nn.Sequential(
            nn.Conv2d(256 + 4 * 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x_in):
        xd, fea_list = self.backbone(x_in)
        xd = self.dam(xd)
        xd = self.aspp(xd)
        xm = self.mfpn([fea_list[i] for i in self.fpn_index])
        x_out = torch.cat([xm, F.interpolate(xd, xm.shape[2:])], dim=1)
        x_out = self.classifier(x_out)
        x_out = F.interpolate(x_out, x_in.shape[2:])
        return x_out
    
