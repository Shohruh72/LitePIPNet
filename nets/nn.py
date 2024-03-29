import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, inp, r=0.0625):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(inp, int(inp * r), 1, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(inp * r), inp, 1, 1, bias=True),
            nn.Sigmoid())

    def forward(self, x):
        b, c, h, w = x.size()
        out = self.avg_pool(x)
        out = self.se(out)
        return x * out.view(-1, c, 1, 1)


class MobileOneBlock(nn.Module):
    def __init__(self, inp, oup, k, s=1, p=0, d=1, g=1, inf_mode=False, se=False, num_conv=1):
        super(MobileOneBlock, self).__init__()
        self.activation = nn.ReLU()
        self.inf_mode = inf_mode
        self.g = g
        self.s = s
        self.k = k
        self.inp = inp
        self.oup = oup
        self.num_conv = num_conv

        self.se = SEBlock(oup) if se else nn.Identity()

        if inf_mode:
            self.reparam_conv = nn.Conv2d(inp, oup, k, s, p, d, g, bias=True)
        else:
            rbr_conv = list()
            self.rbr_skip = nn.BatchNorm2d(inp) if oup == inp and s == 1 else None

            for _ in range(self.num_conv):
                rbr_conv.append(self._conv_bn(k, p))
            self.rbr_conv = nn.ModuleList(rbr_conv)
            self.rbr_scale = self._conv_bn(k=1, p=0) if k > 1 else None

    def forward(self, x):
        if self.inf_mode:
            return self.activation(self.se(self.reparam_conv(x)))

        identity_out = self.rbr_skip(x) if self.rbr_skip is not None else 0
        scale_out = self.rbr_scale(x) if self.rbr_scale is not None else 0
        out = scale_out + identity_out
        for ix in range(self.num_conv):
            out += self.rbr_conv[ix](x)

        return self.activation(self.se(out))

    def re_parameterize(self):
        if self.inf_mode:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(in_channels=self.rbr_conv[0].conv.in_channels,
                                      out_channels=self.rbr_conv[0].conv.out_channels,
                                      kernel_size=self.rbr_conv[0].conv.kernel_size,
                                      stride=self.rbr_conv[0].conv.stride,
                                      padding=self.rbr_conv[0].conv.padding,
                                      dilation=self.rbr_conv[0].conv.dilation,
                                      groups=self.rbr_conv[0].conv.groups,
                                      bias=True)
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_conv')
        self.__delattr__('rbr_scale')
        if hasattr(self, 'rbr_skip'):
            self.__delattr__('rbr_skip')

        self.inf_mode = True

    def _get_kernel_bias(self):
        b_conv = 0
        k_conv = 0
        b_scale = 0
        k_scale = 0
        b_identity = 0
        k_identity = 0

        if self.rbr_scale is not None:
            pad = self.k // 2
            k_scale, b_scale = self._fuse_bn_tensor(self.rbr_scale)
            k_scale = nn.functional.pad(k_scale, [pad, pad, pad, pad])

        if self.rbr_skip is not None:
            k_identity, b_identity = self._fuse_bn_tensor(self.rbr_skip)

        for ix in range(self.num_conv):
            _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv[ix])
            k_conv += _kernel
            b_conv += _bias

        kernel_final = k_conv + k_scale + k_identity
        bias_final = b_conv + b_scale + b_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(self, branch):
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.inp // self.g
                kernel_value = torch.zeros((self.inp, input_dim, self.k, self.k),
                                           dtype=branch.weight.dtype, device=branch.weight.device)
                for i in range(self.inp):
                    kernel_value[i, i % input_dim, self.k // 2, self.k // 2] = 1
                self.id_tensor = kernel_value

            kernel = self.id_tensor
            running_var = branch.running_var
            running_mean = branch.running_mean
            gamma, beta, eps = branch.weight, branch.bias, branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self, k, p):
        mod_list = nn.Sequential()
        mod_list.add_module('conv', nn.Conv2d(self.inp, self.oup, k, self.s, p, groups=self.g, bias=False))
        mod_list.add_module('bn', nn.BatchNorm2d(self.oup))
        return mod_list


class MobileOne(nn.Module):
    def __init__(self, num_cls=1000, width=None, inf_mode=False, use_se=False, num_conv=1):
        super().__init__()

        assert len(width) == 4
        self.use_se = use_se
        self.inf_mode = inf_mode
        self.num_conv = num_conv
        self.in_planes = min(64, int(64 * width[0]))

        self.cur_layer_idx = 1

        self.stage0 = MobileOneBlock(3, self.in_planes, 3, 2, 1, inf_mode=self.inf_mode)
        self.stage1 = self._make_stage(int(64 * width[0]), 2, 0)
        self.stage2 = self._make_stage(int(128 * width[1]), 8, 0)
        self.stage3 = self._make_stage(int(256 * width[2]), 10, int(10 // 2) if use_se else 0)
        self.stage4 = self._make_stage(int(512 * width[3]), 1, 1 if use_se else 0)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(int(512 * width[3]), num_cls)

    def _make_stage(self, planes, num_blocks, se_blocks):
        blocks = []
        for ix, stride in enumerate([2] + [1] * (num_blocks - 1)):
            use_se = False
            if se_blocks > num_blocks:
                raise ValueError("Number of SE blocks cannot exceed number of layers.")

            if ix >= (num_blocks - se_blocks):
                use_se = True

            blocks.append(MobileOneBlock(self.in_planes, self.in_planes, k=3, s=stride, p=1, g=self.in_planes,
                                         inf_mode=self.inf_mode, se=use_se, num_conv=self.num_conv))
            blocks.append(MobileOneBlock(self.in_planes, planes, k=1, s=1, p=0, g=1, inf_mode=self.inf_mode, se=use_se,
                                         num_conv=self.num_conv))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def re_parameterize_model(model):
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 're_parameterize'):
            module.re_parameterize()
    return model


def mobile_one_s0():
    weight_path = Path('./outputs/weights/s0.pth.tar')
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    if not weight_path.exists():
        torch.hub.download_url_to_file(
            'https://docs-assets.developer.apple.com/ml-research/datasets/mobileone/mobileone_s0_unfused.pth.tar',
            str(weight_path))

    return MobileOne(width=(0.75, 1.0, 1.0, 2.0), inf_mode=False, num_conv=4)


def mobile_one_s1():
    weight_path = Path('./outputs/weights/s1.pth.tar')
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    if not weight_path.exists():
        torch.hub.download_url_to_file(
            'https://docs-assets.developer.apple.com/ml-research/datasets/mobileone/mobileone_s1_unfused.pth.tar',
            str(weight_path))
    return MobileOne(width=(1.5, 1.5, 2.0, 2.5))


def mobile_one_s2():
    weight_path = Path('./outputs/weights/s2.pth.tar')
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    if not weight_path.exists():
        torch.hub.download_url_to_file(
            'https://docs-assets.developer.apple.com/ml-research/datasets/mobileone/mobileone_s2_unfused.pth.tar',
            str(weight_path))

    return MobileOne(width=(1.5, 2.0, 2.5, 4.0))


def mobile_one_s3():
    weight_path = Path('./outputs/weights/s3.pth.tar')
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    if not weight_path.exists():
        torch.hub.download_url_to_file(
            'https://docs-assets.developer.apple.com/ml-research/datasets/mobileone/mobileone_s3_unfused.pth.tar',
            str(weight_path))

    return MobileOne(width=(2.0, 2.5, 3.0, 4.0))


def mobile_one_s4():
    weight_path = Path('./outputs/weights/s4.pth.tar')
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    if not weight_path.exists():
        torch.hub.download_url_to_file(
            'https://docs-assets.developer.apple.com/ml-research/datasets/mobileone/mobileone_s4_unfused.pth.tar',
            str(weight_path))
    return MobileOne(width=(3.0, 3.5, 3.5, 4.0), use_se=True)


class PIPNet(nn.Module):
    def __init__(self,
                 params,
                 model_name,
                 for_training,
                 mean_indices, reverse_index1, reverse_index2, max_len):
        super(PIPNet, self).__init__()
        model_selector = {
            's0': mobile_one_s0,
            's1': mobile_one_s1,
            's2': mobile_one_s2,
            's3': mobile_one_s3,
            's4': mobile_one_s4,
        }

        if model_name in model_selector:
            backbone = model_selector[model_name]()
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

        if for_training:
            backbone.load_state_dict(torch.load(f'outputs/weights/{model_name}.pth.tar'))

        self.layer0 = backbone.stage0
        self.layer1 = backbone.stage1
        self.layer2 = backbone.stage2
        self.layer3 = backbone.stage3
        self.layer4 = backbone.stage4

        self.params = params
        self.max_len = max_len
        self.mean_indices = mean_indices
        self.reverse_index1 = reverse_index1
        self.reverse_index2 = reverse_index2

        self.cls_layer = nn.Conv2d(self.layer4[-1].oup, params['num_lms'], 1, 1, 0)
        self.x_layer = nn.Conv2d(self.layer4[-1].oup, params['num_lms'], 1, 1, 0)
        self.y_layer = nn.Conv2d(self.layer4[-1].oup, params['num_lms'], 1, 1, 0)
        self.nb_x_layer = nn.Conv2d(self.layer4[-1].oup, params['num_nb'] * params['num_lms'], 1, 1, 0)
        self.nb_y_layer = nn.Conv2d(self.layer4[-1].oup, params['num_nb'] * params['num_lms'], 1, 1, 0)

        nn.init.normal_(self.cls_layer.weight, std=0.001)
        if self.cls_layer.bias is not None:
            nn.init.constant_(self.cls_layer.bias, 0)

        nn.init.normal_(self.x_layer.weight, std=0.001)
        if self.x_layer.bias is not None:
            nn.init.constant_(self.x_layer.bias, 0)

        nn.init.normal_(self.y_layer.weight, std=0.001)
        if self.y_layer.bias is not None:
            nn.init.constant_(self.y_layer.bias, 0)

        nn.init.normal_(self.nb_x_layer.weight, std=0.001)
        if self.nb_x_layer.bias is not None:
            nn.init.constant_(self.nb_x_layer.bias, 0)

        nn.init.normal_(self.nb_y_layer.weight, std=0.001)
        if self.nb_y_layer.bias is not None:
            nn.init.constant_(self.nb_y_layer.bias, 0)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        out_cls = self.cls_layer(x)
        out_x = self.x_layer(x)
        out_y = self.y_layer(x)
        out_nb_x = self.nb_x_layer(x)
        out_nb_y = self.nb_y_layer(x)
        if self.training:
            return out_cls, out_x, out_y, out_nb_x, out_nb_y

        b, ch, h, w = out_cls.size()
        assert b == 1

        out_cls = out_cls.view(b * ch, -1)
        max_ids = torch.argmax(out_cls, 1)
        max_cls = torch.max(out_cls, 1)[0]
        max_ids = max_ids.view(-1, 1)
        max_ids_nb = max_ids.repeat(1, self.params['num_nb']).view(-1, 1)

        out_x = out_x.view(b * ch, -1)
        out_x_select = torch.gather(out_x, 1, max_ids)
        out_x_select = out_x_select.squeeze(1)
        out_y = out_y.view(b * ch, -1)
        out_y_select = torch.gather(out_y, 1, max_ids)
        out_y_select = out_y_select.squeeze(1)

        out_nb_x = out_nb_x.view(b * self.params['num_nb'] * ch, -1)
        out_nb_x_select = torch.gather(out_nb_x, 1, max_ids_nb)
        out_nb_x_select = out_nb_x_select.squeeze(1).view(-1, self.params['num_nb'])
        out_nb_y = out_nb_y.view(b * self.params['num_nb'] * ch, -1)
        out_nb_y_select = torch.gather(out_nb_y, 1, max_ids_nb)
        out_nb_y_select = out_nb_y_select.squeeze(1).view(-1, self.params['num_nb'])

        tmp_x = (max_ids % w).view(-1, 1).float() + out_x_select.view(-1, 1)
        tmp_y = (max_ids // w).view(-1, 1).float() + out_y_select.view(-1, 1)
        tmp_x /= 1.0 * self.params['input_size'] / self.params['stride']
        tmp_y /= 1.0 * self.params['input_size'] / self.params['stride']

        tmp_nb_x = (max_ids % w).view(-1, 1).float() + out_nb_x_select
        tmp_nb_y = (max_ids // w).view(-1, 1).float() + out_nb_y_select
        tmp_nb_x = tmp_nb_x.view(-1, self.params['num_nb'])
        tmp_nb_y = tmp_nb_y.view(-1, self.params['num_nb'])
        tmp_nb_x /= 1.0 * self.params['input_size'] / self.params['stride']
        tmp_nb_y /= 1.0 * self.params['input_size'] / self.params['stride']

        lms_pred = torch.cat((tmp_x, tmp_y), dim=1).flatten()
        tmp_nb_x = tmp_nb_x[self.reverse_index1, self.reverse_index2].view(self.params['num_lms'], self.max_len)
        tmp_nb_y = tmp_nb_y[self.reverse_index1, self.reverse_index2].view(self.params['num_lms'], self.max_len)
        tmp_x = torch.mean(torch.cat((tmp_x, tmp_nb_x), dim=1), dim=1).view(-1, 1)
        tmp_y = torch.mean(torch.cat((tmp_y, tmp_nb_y), dim=1), dim=1).view(-1, 1)
        lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1)
        return torch.flatten(lms_pred_merge)
