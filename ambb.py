import torch.nn as nn
import torch
from .dbb_transforms import *


def conv_bn(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                   padding_mode='zeros'):
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                           stride=stride, padding=padding, dilation=dilation, groups=groups,
                           bias=False, padding_mode=padding_mode)  # note: bias=False
    bn_layer = nn.BatchNorm2d(num_features=out_channels, affine=True)
    se = nn.Sequential()
    se.add_module('conv', conv_layer)
    se.add_module('bn', bn_layer)
    return se


class AMBB_success(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 deploy=False, nonlinear=None):
        super(AMBB, self).__init__()

        self.deploy = deploy

        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear

        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.groups = groups
        assert padding == kernel_size // 2

        if deploy:
            self.tdb_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True)
        else:
            self.duplicate1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups)

            self.duplicate2 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups)

            self.duplicate3 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups)

            if padding - kernel_size // 2 >= 0:
                #   Common use case. E.g., k=3, p=1 or k=5, p=2
                self.crop = 0
                #   Compared to the KxK layer, the padding of the 1xK layer and Kx1 layer should be adjust to align the sliding windows (Fig 2 in the paper)
                hor_padding = [padding - kernel_size // 2, padding]
                ver_padding = [padding, padding - kernel_size // 2]
            else:
                #   A negative "padding" (padding - kernel_size//2 < 0, which is not a common use case) is cropping.
                #   Since nn.Conv2d does not support negative padding, we implement it manually
                self.crop = kernel_size // 2 - padding
                hor_padding = [0, padding]
                ver_padding = [padding, 0]

            self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, 1),
                                      stride=stride,
                                      padding=ver_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode="zeros")

            self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                                      stride=stride,
                                      padding=hor_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode="zeros")

            self.ver_bn = nn.BatchNorm2d(num_features=out_channels, affine=True)
            self.hor_bn = nn.BatchNorm2d(num_features=out_channels, affine=True)

    def _fuse_bn_tensor(self, conv, bn):
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return conv.weight * t, bn.bias - bn.running_mean * bn.weight / std

    def _add_to_square_kernel(self, square_kernel, asym_kernel):
        asym_h = asym_kernel.size(2)
        asym_w = asym_kernel.size(3)
        square_h = square_kernel.size(2)
        square_w = square_kernel.size(3)
        square_kernel[:, :, square_h // 2 - asym_h // 2: square_h // 2 - asym_h // 2 + asym_h,
                square_w // 2 - asym_w // 2: square_w // 2 - asym_w // 2 + asym_w] += asym_kernel

    def get_equivalent_kernel_bias(self):
        dup1_k, dup1_b = self._fuse_bn_tensor(self.duplicate1.conv, self.duplicate1.bn)
        dup2_k, dup2_b = self._fuse_bn_tensor(self.duplicate2.conv, self.duplicate2.bn)
        dup3_k, dup3_b = self._fuse_bn_tensor(self.duplicate3.conv, self.duplicate3.bn)
        if hasattr(self, "ver_conv") and hasattr(self, "ver_bn"):
            ver_k, ver_b = transI_fusebn(self.ver_conv.weight, self.ver_bn)

        if hasattr(self, "hor_conv") and hasattr(self, "hor_bn"):
            hor_k, hor_b = transI_fusebn(self.hor_conv.weight, self.hor_bn)
        k_origin = dup1_k + dup2_k + dup3_k
        self._add_to_square_kernel(k_origin, hor_k)
        self._add_to_square_kernel(k_origin, ver_k)
        return k_origin, dup1_b + dup2_b + dup3_b + ver_b + hor_b

    def switch_to_deploy(self):
        deploy_k, deploy_b = self.get_equivalent_kernel_bias()
        self.deploy = True
        self.tdb_reparam = nn.Conv2d(in_channels=self.duplicate1.conv.in_channels,
                                     out_channels=self.duplicate1.conv.out_channels,
                                     kernel_size=self.duplicate1.conv.kernel_size,
                                     stride=self.duplicate1.conv.stride,
                                     padding=self.duplicate1.conv.padding,
                                     dilation=self.duplicate1.conv.dilation,
                                     groups=self.duplicate1.conv.groups)
        self.__delattr__('duplicate1')
        self.__delattr__('duplicate2')
        self.__delattr__('duplicate3')
        self.__delattr__('ver_conv')
        self.__delattr__('ver_bn')
        self.__delattr__('hor_conv')
        self.__delattr__('hor_bn')
        self.tdb_reparam.weight.data = deploy_k
        self.tdb_reparam.bias.data = deploy_b

    def forward(self, inputs):
        if hasattr(self, 'tdb_reparam'):
            return self.nonlinear(self.tdb_reparam(inputs))

        return self.nonlinear(self.duplicate1(inputs) + self.duplicate2(inputs) + self.duplicate3(inputs) + \
                              self.ver_bn(self.ver_conv(inputs)) + self.hor_bn(self.hor_conv(inputs)))

# 在AMBB基础上加1x1操作
class AMBB_2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 deploy=False, nonlinear=None):
        super(AMBB, self).__init__()  # 修正类名继承
        self.deploy = deploy
        self.nonlinear = nonlinear if nonlinear is not None else nn.Identity()
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.groups = groups
        self.dilation = dilation
        self.stride = stride
        self.in_channels = in_channels
        self.padding = padding
        assert padding == kernel_size // 2, "Padding must be kernel_size//2 for size matching"

        if deploy:
            # 部署模式：单K×K卷积
            self.tdb_reparam = nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True
            )
        else:
            # 分支1：1×1 Conv+BN → K×K Conv+BN（串联结构）
            self.duplicate1_1x1 = conv_bn(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                stride=stride, padding=0, dilation=dilation, groups=groups
            )
            self.duplicate1 = conv_bn(
                in_channels=out_channels,  # 输入通道=1×1的输出通道
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,  # stride由1×1卷积完成
                padding=padding, dilation=dilation, groups=groups
            )

            # 分支2：1×1 Conv+BN → K×K Conv+BN（串联结构）
            self.duplicate2_1x1 = conv_bn(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                stride=stride, padding=0, dilation=dilation, groups=groups
            )
            self.duplicate2 = conv_bn(
                in_channels=out_channels,  # 输入通道=1×1的输出通道
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,  # stride由1×1卷积完成
                padding=padding, dilation=dilation, groups=groups
            )

            # 分支3：1×1 Conv+BN → K×K Conv+BN（串联结构）
            self.duplicate3_1x1 = conv_bn(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                stride=stride, padding=0, dilation=dilation, groups=groups
            )
            self.duplicate3 = conv_bn(
                in_channels=out_channels,  # 输入通道=1×1的输出通道
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,  # stride由1×1卷积完成
                padding=padding, dilation=dilation, groups=groups
            )

            # 非对称卷积分支：1×1 Conv+BN → K×1 Conv（垂直分支）
            self.ver_1x1 = conv_bn(  # 补充缺失的ver_1x1层
                in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                stride=stride, padding=0, dilation=dilation, groups=groups
            )
            ver_padding = (kernel_size // 2, 0)  # K×1卷积的padding
            self.ver_conv = nn.Conv2d(
                in_channels=out_channels,  # 输入通道=1×1的输出通道
                out_channels=out_channels,
                kernel_size=(kernel_size, 1),
                stride=1,  # stride由前面的1×1卷积完成
                padding=ver_padding, dilation=dilation, groups=groups,
                bias=False, padding_mode="zeros"
            )
            self.ver_bn = nn.BatchNorm2d(num_features=out_channels, affine=True)

            # 非对称卷积分支：1×1 Conv+BN → 1×K Conv（水平分支）
            self.hor_1x1 = conv_bn(  # 补充缺失的hor_1x1层
                in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                stride=stride, padding=0, dilation=dilation, groups=groups
            )
            hor_padding = (0, kernel_size // 2)  # 1×K卷积的padding
            self.hor_conv = nn.Conv2d(
                in_channels=out_channels,  # 输入通道=1×1的输出通道
                out_channels=out_channels,
                kernel_size=(1, kernel_size),
                stride=1,  # stride由前面的1×1卷积完成
                padding=hor_padding, dilation=dilation, groups=groups,
                bias=False, padding_mode="zeros"
            )
            self.hor_bn = nn.BatchNorm2d(num_features=out_channels, affine=True)

            # 裁剪标记（典型场景下为0）
            self.crop = 0

    def _fuse_bn_tensor(self, conv, bn):
        """融合单个Conv+BN的核和偏置"""
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return conv.weight * t, bn.bias - bn.running_mean * bn.weight / std

    def _add_to_square_kernel(self, square_kernel, asym_kernel):
        """将非方形核叠加到方形核的中心（非in-place操作）"""
        square_kernel = square_kernel.clone()
        asym_h, asym_w = asym_kernel.size(2), asym_kernel.size(3)
        square_h, square_w = square_kernel.size(2), square_kernel.size(3)
        h_start = square_h // 2 - asym_h // 2
        w_start = square_w // 2 - asym_w // 2
        square_kernel[:, :, h_start:h_start + asym_h, w_start:w_start + asym_w] += asym_kernel
        return square_kernel

    def get_equivalent_kernel_bias(self):
        """融合所有分支为单个K×K卷积的核和偏置（修复所有逻辑错误）"""
        device = next(self.parameters()).device
        K = self.kernel_size

        # --------------------------- 分支1：1×1 → K×K 串联融合 ---------------------------
        k1_1x1, b1_1x1 = self._fuse_bn_tensor(self.duplicate1_1x1.conv, self.duplicate1_1x1.bn)
        k1_kxk, b1_kxk = self._fuse_bn_tensor(self.duplicate1.conv, self.duplicate1.bn)
        dup1_k, dup1_b = transIII_1x1_kxk(k1_1x1, b1_1x1, k1_kxk, b1_kxk, self.groups)

        # --------------------------- 分支2：1×1 → K×K 串联融合 ---------------------------
        k2_1x1, b2_1x1 = self._fuse_bn_tensor(self.duplicate2_1x1.conv, self.duplicate2_1x1.bn)
        k2_kxk, b2_kxk = self._fuse_bn_tensor(self.duplicate2.conv, self.duplicate2.bn)
        dup2_k, dup2_b = transIII_1x1_kxk(k2_1x1, b2_1x1, k2_kxk, b2_kxk, self.groups)

        # --------------------------- 分支3：1×1 → K×K 串联融合 ---------------------------
        k3_1x1, b3_1x1 = self._fuse_bn_tensor(self.duplicate3_1x1.conv, self.duplicate3_1x1.bn)
        k3_kxk, b3_kxk = self._fuse_bn_tensor(self.duplicate3.conv, self.duplicate3.bn)
        dup3_k, dup3_b = transIII_1x1_kxk(k3_1x1, b3_1x1, k3_kxk, b3_kxk, self.groups)

        # --------------------------- 垂直分支：1×1 → K×1 串联融合 ---------------------------
        # 融合1×1 Conv+BN
        ver_1x1_k, ver_1x1_b = self._fuse_bn_tensor(self.ver_1x1.conv, self.ver_1x1.bn)
        # 融合K×1 Conv+BN
        ver_conv_k, ver_conv_b = transI_fusebn(self.ver_conv.weight, self.ver_bn)
        # 串联融合
        ver_k, ver_b = transIII_1x1_kxk(ver_1x1_k, ver_1x1_b, ver_conv_k, ver_conv_b, self.groups)
        # 将K×1核pad为K×K（指定设备）
        pad_w = (K - ver_k.size(3)) // 2
        ver_k_pad = F.pad(ver_k, (pad_w, pad_w, 0, 0))

        # --------------------------- 水平分支：1×1 → 1×K 串联融合 ---------------------------
        # 融合1×1 Conv+BN
        hor_1x1_k, hor_1x1_b = self._fuse_bn_tensor(self.hor_1x1.conv, self.hor_1x1.bn)
        # 融合1×K Conv+BN
        hor_conv_k, hor_conv_b = transI_fusebn(self.hor_conv.weight, self.hor_bn)
        # 串联融合
        hor_k, hor_b = transIII_1x1_kxk(hor_1x1_k, hor_1x1_b, hor_conv_k, hor_conv_b, self.groups)
        # 将1×K核pad为K×K（指定设备）
        pad_h = (K - hor_k.size(2)) // 2
        hor_k_pad = F.pad(hor_k, (0, 0, pad_h, pad_h))

        # --------------------------- 所有分支核和偏置相加 ---------------------------
        k_total = dup1_k + dup2_k + dup3_k + ver_k_pad + hor_k_pad
        b_total = dup1_b + dup2_b + dup3_b + ver_b + hor_b

        return k_total, b_total

    def switch_to_deploy(self):
        """切换到部署模式，删除所有训练分支"""
        deploy_k, deploy_b = self.get_equivalent_kernel_bias()
        self.deploy = True

        # 创建部署用的K×K卷积
        self.tdb_reparam = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=True
        )

        # 赋值融合后的核和偏置（禁用梯度）
        with torch.no_grad():
            self.tdb_reparam.weight.copy_(deploy_k)
            self.tdb_reparam.bias.copy_(deploy_b)

        # 删除所有训练分支（完整列表）
        del_attrs = [
            'duplicate1_1x1', 'duplicate1', 'duplicate2_1x1', 'duplicate2',
            'duplicate3_1x1', 'duplicate3', 'ver_1x1', 'ver_conv', 'ver_bn',
            'hor_1x1', 'hor_conv', 'hor_bn'
        ]
        for attr in del_attrs:
            if hasattr(self, attr):
                self.__delattr__(attr)

    def forward(self, inputs):
        if self.deploy and hasattr(self, 'tdb_reparam'):
            return self.nonlinear(self.tdb_reparam(inputs))

        # 训练模式：各分支前向计算（修正所有输入错误）
        out1 = self.duplicate1(self.duplicate1_1x1(inputs))
        out2 = self.duplicate2(self.duplicate2_1x1(inputs))
        out3 = self.duplicate3(self.duplicate3_1x1(inputs))  # 修正分支3的输入

        # 垂直分支：1×1 → K×1 → BN（完整前向）
        ver_out = self.ver_bn(self.ver_conv(self.ver_1x1(inputs)))
        # 水平分支：1×1 → 1×K → BN（完整前向）
        hor_out = self.hor_bn(self.hor_conv(self.hor_1x1(inputs)))

        # 裁剪（典型场景下为0）
        if self.crop > 0:
            ver_out = ver_out[:, :, self.crop:-self.crop, :]
            hor_out = hor_out[:, :, :, self.crop:-self.crop]

        # 总输出
        total_out = out1 + out2 + out3 + ver_out + hor_out
        return self.nonlinear(total_out)


class AMBB_3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 deploy=False, nonlinear=None):
        super(AMBB, self).__init__()

        self.deploy = deploy
        self.in_channels = in_channels  # 保存原始输入通道
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # 非线性层
        self.nonlinear = nonlinear if nonlinear is not None else nn.Identity()

        assert padding == kernel_size // 2, "Padding must be kernel_size//2 for size matching"

        if deploy:
            # 部署模式：单K×K卷积（带bias）
            self.tdb_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True
            )
        else:
            # 分支1：1×1 Conv+BN → K×K Conv+BN（串联结构）
            self.duplicate1_1x1 = conv_bn(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                stride=stride, padding=0, dilation=dilation, groups=groups
            )
            self.duplicate1 = conv_bn(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding, dilation=dilation, groups=groups
            )

            # 分支2：1×1 Conv+BN → K×K Conv+BN（串联结构）
            self.duplicate2_1x1 = conv_bn(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                stride=stride, padding=0, dilation=dilation, groups=groups
            )
            self.duplicate2 = conv_bn(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding, dilation=dilation, groups=groups
            )

            # 分支3：1×1 Conv+BN → K×K Conv+BN（串联结构）
            self.duplicate3_1x1 = conv_bn(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                stride=stride, padding=0, dilation=dilation, groups=groups
            )
            self.duplicate3 = conv_bn(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding, dilation=dilation, groups=groups
            )

            # 修正非对称卷积的padding计算（符合PyTorch规范）
            if padding - kernel_size // 2 >= 0:
                self.crop = 0
                # PyTorch的padding是(高度padding, 宽度padding)
                ver_padding = (padding, padding - kernel_size // 2)  # (k,1)卷积的padding
                hor_padding = (padding - kernel_size // 2, padding)  # (1,k)卷积的padding
            else:
                self.crop = kernel_size // 2 - padding
                ver_padding = (padding, 0)
                hor_padding = (0, padding)

            # 垂直卷积：(kernel_size, 1)
            self.ver_conv = nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=(kernel_size, 1),
                stride=stride,
                padding=ver_padding,
                dilation=dilation,
                groups=groups,
                bias=False,
                padding_mode="zeros"
            )

            # 水平卷积：(1, kernel_size)
            self.hor_conv = nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=(1, kernel_size),
                stride=stride,
                padding=hor_padding,
                dilation=dilation,
                groups=groups,
                bias=False,
                padding_mode="zeros"
            )

            # BN层
            self.ver_bn = nn.BatchNorm2d(num_features=out_channels, affine=True)
            self.hor_bn = nn.BatchNorm2d(num_features=out_channels, affine=True)

    def _fuse_bn_tensor(self, conv, bn):
        """融合单个Conv+BN的核与偏置"""
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return conv.weight * t, bn.bias - bn.running_mean * bn.weight / std

    def _add_to_square_kernel(self, square_kernel, asym_kernel):
        """将非方形核叠加到方形核（非原地操作，返回新核）"""
        square_kernel = square_kernel.clone()  # 避免原地修改
        asym_h, asym_w = asym_kernel.size(2), asym_kernel.size(3)
        square_h, square_w = square_kernel.size(2), square_kernel.size(3)

        # 计算中心偏移
        h_start = square_h // 2 - asym_h // 2
        w_start = square_w // 2 - asym_w // 2

        # 叠加核
        square_kernel[:, :, h_start:h_start + asym_h, w_start:w_start + asym_w] += asym_kernel
        return square_kernel  # 返回新核

    def get_equivalent_kernel_bias(self):
        """计算等效的K×K核和偏置"""
        device = next(self.parameters()).device  # 获取模型设备
        K = self.kernel_size

        # --------------------------- 融合三个K×K分支 ---------------------------
        # 分支1：1×1→K×K
        k1_1x1, b1_1x1 = self._fuse_bn_tensor(self.duplicate1_1x1.conv, self.duplicate1_1x1.bn)
        dup1_k, dup1_b = self._fuse_bn_tensor(self.duplicate1.conv, self.duplicate1.bn)
        dup1_k, dup1_b = transIII_1x1_kxk(k1_1x1, b1_1x1, dup1_k, dup1_b, self.groups)

        # 分支2：1×1→K×K
        k2_1x1, b2_1x1 = self._fuse_bn_tensor(self.duplicate2_1x1.conv, self.duplicate2_1x1.bn)
        dup2_k, dup2_b = self._fuse_bn_tensor(self.duplicate2.conv, self.duplicate2.bn)
        dup2_k, dup2_b = transIII_1x1_kxk(k2_1x1, b2_1x1, dup2_k, dup2_b, self.groups)

        # 分支3：1×1→K×K
        k3_1x1, b3_1x1 = self._fuse_bn_tensor(self.duplicate3_1x1.conv, self.duplicate3_1x1.bn)
        dup3_k, dup3_b = self._fuse_bn_tensor(self.duplicate3.conv, self.duplicate3.bn)
        dup3_k, dup3_b = transIII_1x1_kxk(k3_1x1, b3_1x1, dup3_k, dup3_b, self.groups)

        # --------------------------- 融合非对称卷积分支 ---------------------------
        ver_k, ver_b = 0, 0
        hor_k, hor_b = 0, 0

        if hasattr(self, "ver_conv") and hasattr(self, "ver_bn"):
            ver_k, ver_b = transI_fusebn(self.ver_conv.weight, self.ver_bn)
            ver_k = ver_k.to(device)
            ver_b = ver_b.to(device)

        if hasattr(self, "hor_conv") and hasattr(self, "hor_bn"):
            hor_k, hor_b = transI_fusebn(self.hor_conv.weight, self.hor_bn)
            hor_k = hor_k.to(device)
            hor_b = hor_b.to(device)

        # --------------------------- 叠加所有核 ---------------------------
        k_origin = dup1_k + dup2_k + dup3_k
        k_origin = self._add_to_square_kernel(k_origin, hor_k)
        k_origin = self._add_to_square_kernel(k_origin, ver_k)

        # 叠加所有偏置
        b_total = dup1_b + dup2_b + dup3_b + ver_b + hor_b

        return k_origin.to(device), b_total.to(device)

    def switch_to_deploy(self):
        """切换到部署模式"""
        if self.deploy:
            return

        # 计算等效核和偏置
        deploy_k, deploy_b = self.get_equivalent_kernel_bias()

        # 创建部署用的卷积层（修正in_channels和bias=True）
        self.tdb_reparam = nn.Conv2d(
            in_channels=self.in_channels,  # 原始输入通道
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=True  # 启用bias
        )

        # 赋值核和偏置
        with torch.no_grad():
            self.tdb_reparam.weight.data = deploy_k
            self.tdb_reparam.bias.data = deploy_b

        # 删除所有训练层（完整列表）
        del_attrs = [
            'duplicate1_1x1', 'duplicate1', 'duplicate2_1x1', 'duplicate2',
            'duplicate3_1x1', 'duplicate3', 'ver_conv', 'ver_bn', 'hor_conv', 'hor_bn'
        ]
        for attr in del_attrs:
            if hasattr(self, attr):
                self.__delattr__(attr)

        self.deploy = True

    def forward(self, inputs):
        """前向传播"""
        if self.deploy:
            return self.nonlinear(self.tdb_reparam(inputs))
        print('++++++++++',inputs.shape)
        # 训练模式：多分支求和
        out1 = self.duplicate1(self.duplicate1_1x1(inputs))
        out2 = self.duplicate2(self.duplicate2_1x1(inputs))
        out3 = self.duplicate3(self.duplicate3_1x1(inputs))
        ver_out = self.ver_bn(self.ver_conv(inputs))
        hor_out = self.hor_bn(self.hor_conv(inputs))

        return self.nonlinear(out1 + out2 + out3 + ver_out + hor_out)


class AMBB_3_1_3_success(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 deploy=False, nonlinear=None):
        super(AMBB_3_1_3_success, self).__init__()
        self.deploy = deploy
        self.nonlinear = nonlinear if nonlinear is not None else nn.Identity()
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.groups = groups
        self.dilation = dilation
        self.stride = stride
        assert padding == kernel_size // 2, "Padding must be kernel_size//2 for size matching"

        if deploy:
            # 部署模式：单K×K卷积
            self.tdb_reparam = nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True
            )
        else:
            # 分支1：1×1 Conv + BN（点卷积分支）
            self.duplicate1 = conv_bn(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                stride=stride, padding=0, dilation=dilation, groups=groups
            )

            # 分支2：K×K Conv + BN（主空间分支）
            self.duplicate2 = conv_bn(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups
            )

            # 分支3：K×K Conv + BN（辅助空间分支）
            self.duplicate3 = conv_bn(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups
            )

            # 非对称卷积分支的Padding计算（修复格式错误）
            # 对于K×1卷积（ver_conv）：高度方向padding=K//2，宽度方向padding=0
            # 对于1×K卷积（hor_conv）：高度方向padding=0，宽度方向padding=K//2
            ver_padding = (kernel_size // 2, 0)  # (height_pad, width_pad)
            hor_padding = (0, kernel_size // 2)  # (height_pad, width_pad)

            # 垂直卷积：K×1（如3×1）
            self.ver_conv = nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, 1),
                stride=stride, padding=ver_padding, dilation=dilation, groups=groups,
                bias=False, padding_mode="zeros"
            )

            # 水平卷积：1×K（如1×3）
            self.hor_conv = nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                stride=stride, padding=hor_padding, dilation=dilation, groups=groups,
                bias=False, padding_mode="zeros"
            )

            # 非对称卷积的BN层
            self.ver_bn = nn.BatchNorm2d(num_features=out_channels, affine=True)
            self.hor_bn = nn.BatchNorm2d(num_features=out_channels, affine=True)

            # 裁剪标记（简化逻辑，当前场景下crop=0）
            self.crop = 0

    def _fuse_bn_tensor(self, conv, bn):
        """融合单个Conv+BN的核和偏置"""
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return conv.weight * t, bn.bias - bn.running_mean * bn.weight / std

    def _add_to_square_kernel(self, square_kernel, asym_kernel):
        """将非方形核叠加到方形核的中心（非in-place操作，避免梯度问题）"""
        square_kernel = square_kernel.clone()
        asym_h, asym_w = asym_kernel.size(2), asym_kernel.size(3)
        square_h, square_w = square_kernel.size(2), square_kernel.size(3)

        # 计算中心偏移
        h_start = square_h // 2 - asym_h // 2
        w_start = square_w // 2 - asym_w // 2

        # 叠加核
        square_kernel[:, :, h_start:h_start + asym_h, w_start:w_start + asym_w] += asym_kernel
        return square_kernel

    def _add_to_square_kernel_1x1(self, square_kernel, asym_kernel):
        """将1×1核叠加到方形核中心"""
        return self._add_to_square_kernel(square_kernel, asym_kernel)

    def get_equivalent_kernel_bias(self):
        """融合所有分支为单个K×K卷积的核和偏置"""
        # 融合分支1：1×1 Conv+BN → 扩展为K×K核
        dup1_k_1x1, dup1_b = self._fuse_bn_tensor(self.duplicate1.conv, self.duplicate1.bn)
        dup2_k, dup2_b = self._fuse_bn_tensor(self.duplicate2.conv, self.duplicate2.bn)
        # 初始化K×K核并叠加1×1核
        dup1_k = torch.zeros_like(dup2_k, device=dup2_k.device)
        dup1_k = self._add_to_square_kernel_1x1(dup1_k, dup1_k_1x1)

        # 融合分支3：K×K Conv+BN
        dup3_k, dup3_b = self._fuse_bn_tensor(self.duplicate3.conv, self.duplicate3.bn)

        # 融合非对称卷积分支
        ver_k, ver_b = transI_fusebn(self.ver_conv.weight, self.ver_bn)
        hor_k, hor_b = transI_fusebn(self.hor_conv.weight, self.hor_bn)

        # 所有核叠加
        k_origin = dup1_k + dup2_k + dup3_k
        k_origin = self._add_to_square_kernel(k_origin, hor_k)
        k_origin = self._add_to_square_kernel(k_origin, ver_k)

        # 所有偏置相加
        total_b = dup1_b + dup2_b + dup3_b + ver_b + hor_b

        return k_origin, total_b

    def switch_to_deploy(self):
        """切换到部署模式，删除训练分支"""
        deploy_k, deploy_b = self.get_equivalent_kernel_bias()
        self.deploy = True

        # 创建部署用的K×K卷积（修复核尺寸参数错误）
        self.tdb_reparam = nn.Conv2d(
            in_channels=self.duplicate1.conv.in_channels,
            out_channels=self.duplicate1.conv.out_channels,
            kernel_size=self.kernel_size,  # 改为K×K，而非1×1
            stride=self.stride,
            padding=self.kernel_size // 2,  # 匹配K×K的padding
            dilation=self.dilation,
            groups=self.groups,
            bias=True
        )

        # 赋值融合后的核和偏置
        self.tdb_reparam.weight.data = deploy_k
        self.tdb_reparam.bias.data = deploy_b

        # 删除训练分支
        del_attrs = ['duplicate1', 'duplicate2', 'duplicate3', 'ver_conv', 'ver_bn', 'hor_conv', 'hor_bn']
        for attr in del_attrs:
            if hasattr(self, attr):
                self.__delattr__(attr)

    def forward(self, inputs):
        if self.deploy:
            return self.nonlinear(self.tdb_reparam(inputs))

        # 训练模式：所有分支输出相加
        out1 = self.duplicate1(inputs)
        out2 = self.duplicate2(inputs)
        out3 = self.duplicate3(inputs)

        # 非对称卷积分支（处理裁剪）
        ver_out = self.ver_bn(self.ver_conv(inputs))
        hor_out = self.hor_bn(self.hor_conv(inputs))
        if self.crop > 0:
            ver_out = ver_out[:, :, self.crop:-self.crop, :]
            hor_out = hor_out[:, :, :, self.crop:-self.crop]

        # 总输出
        total_out = out1 + out2 + out3 + ver_out + hor_out
        return self.nonlinear(total_out)



class AMBB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 deploy=False, nonlinear=None):
        super(AMBB, self).__init__()

        self.deploy = deploy
        self.in_channels = in_channels  # 保存原始输入通道
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # 非线性层
        self.nonlinear = nonlinear if nonlinear is not None else nn.Identity()

        assert padding == kernel_size // 2, "Padding must be kernel_size//2 for size matching"

        if deploy:
            # 部署模式：单K×K卷积（带bias）
            self.tdb_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True
            )
        else:
            # 分支1：1×1 Conv+BN → K×K Conv+BN（串联结构）
            self.duplicate1_1x1 = conv_bn(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                stride=1, padding=0, dilation=dilation, groups=groups
            )
            self.duplicate1 = conv_bn(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding, dilation=dilation, groups=groups
            )

            # 分支2：1×1 Conv+BN → K×K Conv+BN（串联结构）
            self.duplicate2_1x1 = conv_bn(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                stride=1, padding=0, dilation=dilation, groups=groups
            )
            self.duplicate2 = conv_bn(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding, dilation=dilation, groups=groups
            )

            # 分支3：1×1 Conv+BN → K×K Conv+BN（串联结构）
            self.duplicate3_1x1 = conv_bn(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                stride=1, padding=0, dilation=dilation, groups=groups
            )
            self.duplicate3 = conv_bn(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding, dilation=dilation, groups=groups
            )

            # 修正非对称卷积的padding计算（符合PyTorch规范）
            if padding - kernel_size // 2 >= 0:
                self.crop = 0
                # PyTorch的padding是(高度padding, 宽度padding)
                ver_padding = (padding, padding - kernel_size // 2)  # (k,1)卷积的padding
                hor_padding = (padding - kernel_size // 2, padding)  # (1,k)卷积的padding
            else:
                self.crop = kernel_size // 2 - padding
                ver_padding = (padding, 0)
                hor_padding = (0, padding)

            # 垂直卷积：(kernel_size, 1)
            self.ver_conv = nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=(kernel_size, 1),
                stride=stride,
                padding=ver_padding,
                dilation=dilation,
                groups=groups,
                bias=False,
                padding_mode="zeros"
            )

            # 水平卷积：(1, kernel_size)
            self.hor_conv = nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=(1, kernel_size),
                stride=stride,
                padding=hor_padding,
                dilation=dilation,
                groups=groups,
                bias=False,
                padding_mode="zeros"
            )

            # BN层
            self.ver_bn = nn.BatchNorm2d(num_features=out_channels, affine=True)
            self.hor_bn = nn.BatchNorm2d(num_features=out_channels, affine=True)

    def _fuse_bn_tensor(self, conv, bn):
        """融合单个Conv+BN的核与偏置"""
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return conv.weight * t, bn.bias - bn.running_mean * bn.weight / std

    def _add_to_square_kernel(self, square_kernel, asym_kernel):
        """将非方形核叠加到方形核（非原地操作，返回新核）"""
        square_kernel = square_kernel.clone()  # 避免原地修改
        asym_h, asym_w = asym_kernel.size(2), asym_kernel.size(3)
        square_h, square_w = square_kernel.size(2), square_kernel.size(3)

        # 计算中心偏移
        h_start = square_h // 2 - asym_h // 2
        w_start = square_w // 2 - asym_w // 2

        # 叠加核
        square_kernel[:, :, h_start:h_start + asym_h, w_start:w_start + asym_w] += asym_kernel
        return square_kernel  # 返回新核

    def get_equivalent_kernel_bias(self):
        """计算等效的K×K核和偏置"""
        device = next(self.parameters()).device  # 获取模型设备
        K = self.kernel_size

        # --------------------------- 融合三个K×K分支 ---------------------------
        # 分支1：1×1→K×K
        k1_1x1, b1_1x1 = self._fuse_bn_tensor(self.duplicate1_1x1.conv, self.duplicate1_1x1.bn)
        dup1_k, dup1_b = self._fuse_bn_tensor(self.duplicate1.conv, self.duplicate1.bn)
        dup1_k, dup1_b = transIII_1x1_kxk(k1_1x1, b1_1x1, dup1_k, dup1_b, self.groups)

        # 分支2：1×1→K×K
        k2_1x1, b2_1x1 = self._fuse_bn_tensor(self.duplicate2_1x1.conv, self.duplicate2_1x1.bn)
        dup2_k, dup2_b = self._fuse_bn_tensor(self.duplicate2.conv, self.duplicate2.bn)
        dup2_k, dup2_b = transIII_1x1_kxk(k2_1x1, b2_1x1, dup2_k, dup2_b, self.groups)

        # 分支3：1×1→K×K
        k3_1x1, b3_1x1 = self._fuse_bn_tensor(self.duplicate3_1x1.conv, self.duplicate3_1x1.bn)
        dup3_k, dup3_b = self._fuse_bn_tensor(self.duplicate3.conv, self.duplicate3.bn)
        dup3_k, dup3_b = transIII_1x1_kxk(k3_1x1, b3_1x1, dup3_k, dup3_b, self.groups)

        # --------------------------- 融合非对称卷积分支 ---------------------------
        ver_k, ver_b = 0, 0
        hor_k, hor_b = 0, 0

        if hasattr(self, "ver_conv") and hasattr(self, "ver_bn"):
            ver_k, ver_b = transI_fusebn(self.ver_conv.weight, self.ver_bn)
            ver_k = ver_k.to(device)
            ver_b = ver_b.to(device)

        if hasattr(self, "hor_conv") and hasattr(self, "hor_bn"):
            hor_k, hor_b = transI_fusebn(self.hor_conv.weight, self.hor_bn)
            hor_k = hor_k.to(device)
            hor_b = hor_b.to(device)

        # --------------------------- 叠加所有核 ---------------------------
        k_origin = dup1_k + dup2_k + dup3_k
        k_origin = self._add_to_square_kernel(k_origin, hor_k)
        k_origin = self._add_to_square_kernel(k_origin, ver_k)

        # 叠加所有偏置
        b_total = dup1_b + dup2_b + dup3_b + ver_b + hor_b

        return k_origin.to(device), b_total.to(device)

    def switch_to_deploy(self):
        """切换到部署模式"""
        if self.deploy:
            return

        # 计算等效核和偏置
        deploy_k, deploy_b = self.get_equivalent_kernel_bias()

        # 创建部署用的卷积层（修正in_channels和bias=True）
        self.tdb_reparam = nn.Conv2d(
            in_channels=self.in_channels,  # 原始输入通道
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=True  # 启用bias
        )

        # 赋值核和偏置
        with torch.no_grad():
            self.tdb_reparam.weight.data = deploy_k
            self.tdb_reparam.bias.data = deploy_b

        # 删除所有训练层（完整列表）
        del_attrs = [
            'duplicate1_1x1', 'duplicate1', 'duplicate2_1x1', 'duplicate2',
            'duplicate3_1x1', 'duplicate3', 'ver_conv', 'ver_bn', 'hor_conv', 'hor_bn'
        ]
        for attr in del_attrs:
            if hasattr(self, attr):
                self.__delattr__(attr)

        self.deploy = True

    def forward(self, inputs):
        """前向传播"""
        if self.deploy:
            return self.nonlinear(self.tdb_reparam(inputs))
        # 训练模式：多分支求和
        out1 = self.duplicate1(self.duplicate1_1x1(inputs))
        out2 = self.duplicate2(self.duplicate2_1x1(inputs))
        out3 = self.duplicate3(self.duplicate3_1x1(inputs))
        ver_out = self.ver_bn(self.ver_conv(inputs))
        hor_out = self.hor_bn(self.hor_conv(inputs))

        return self.nonlinear(out1 + out2 + out3 + ver_out + hor_out)



class AMBB_3_1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 deploy=False, nonlinear=None):
        super(AMBB_3_1, self).__init__()

        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # 激活函数
        self.nonlinear = nonlinear if nonlinear is not None else nn.Identity()

        # 校验padding（保证kxk卷积输入输出尺寸一致）
        assert padding == kernel_size // 2, f"padding应设为{kernel_size//2}，当前为{padding}"

        if deploy:
            # 推理阶段：单个重参数卷积
            self.tdb_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True
            )
        else:
            # 训练阶段：三个并行分支
            # 分支1：kxk卷积 + BN
            self.duplicate1 = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups
            )

            # 分支2：1x1卷积 + BN
            self.duplicate2 = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                dilation=dilation,
                groups=groups
            )

            # 分支3：Identity + BN（先1x1卷积做通道映射，再BN，避免通道不匹配）
            self.identity_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                groups=groups,
                bias=False  # 偏置由BN吸收
            )
            # 初始化identity_conv为单位映射（in=out时）
            if in_channels == out_channels:
                nn.init.eye_(self.identity_conv.weight.data.reshape(out_channels, in_channels))
            self.self_bn = nn.BatchNorm2d(num_features=out_channels, affine=True)

    def _fuse_bn_tensor(self, conv, bn):
        """融合Conv + BN的参数：返回融合后的权重和偏置"""
        with torch.no_grad():
            std = (bn.running_var + bn.eps).sqrt()
            # BN缩放因子：gamma / std
            scale = (bn.weight / std).reshape(-1, 1, 1, 1)
            # 融合后的卷积权重
            fused_weight = conv.weight * scale
            # 融合后的卷积偏置
            fused_bias = bn.bias - (bn.running_mean * bn.weight) / std
        return fused_weight, fused_bias

    def _add_to_square_kernel(self, square_kernel, asym_kernel):
        """将非方形核（如1x1）扩展并叠加到方形核（如3x3）的中心"""
        with torch.no_grad():
            asym_h, asym_w = asym_kernel.size(2), asym_kernel.size(3)
            square_h, square_w = square_kernel.size(2), square_kernel.size(3)
            # 计算中心偏移
            h_start = square_h // 2 - asym_h // 2
            w_start = square_w // 2 - asym_w // 2
            # 叠加到方形核中心
            square_kernel[:, :, h_start:h_start+asym_h, w_start:w_start+asym_w] += asym_kernel
        return square_kernel

    def get_equivalent_kernel_bias(self):
        """计算三个分支融合后的等效卷积核和偏置"""
        with torch.no_grad():
            # 1. 融合分支1：kxk Conv + BN
            k1, b1 = self._fuse_bn_tensor(self.duplicate1.conv, self.duplicate1.bn)

            # 2. 融合分支2：1x1 Conv + BN → 扩展为kxk核
            k2_1x1, b2 = self._fuse_bn_tensor(self.duplicate2.conv, self.duplicate2.bn)
            # 初始化kxk核（和分支1同尺寸）
            k2 = torch.zeros_like(k1)
            # 将1x1核扩展到kxk核的中心
            k2 = self._add_to_square_kernel(k2, k2_1x1)

            # 3. 融合分支3：Identity Conv + BN → 扩展为kxk核
            k3_1x1, b3 = self._fuse_bn_tensor(self.identity_conv, self.self_bn)
            # 初始化kxk核
            k3 = torch.zeros_like(k1)
            # 将Identity的1x1核扩展到kxk核的中心
            k3 = self._add_to_square_kernel(k3, k3_1x1)

            # 4. 三个分支的核和偏置相加（并行分支的重参数化核心）
            deploy_k = k1 + k2 + k3
            deploy_b = b1 + b2 + b3

        return deploy_k, deploy_b

    def switch_to_deploy(self):
        """切换到推理模式：融合分支为单个卷积"""
        if self.deploy:
            return
        # 获取融合后的核和偏置
        deploy_k, deploy_b = self.get_equivalent_kernel_bias()
        # 初始化推理阶段的卷积层
        self.tdb_reparam = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=True
        )
        # 赋值融合后的参数
        self.tdb_reparam.weight.data = deploy_k
        self.tdb_reparam.bias.data = deploy_b
        # 标记为推理模式
        self.deploy = True
        # 删除训练阶段的分支（释放内存）
        del_attrs = ['duplicate1', 'duplicate2', 'identity_conv', 'self_bn']
        for attr in del_attrs:
            if hasattr(self, attr):
                self.__delattr__(attr)

    def forward(self, inputs):
        """前向传播：训练/推理分支自动切换"""
        if self.deploy or hasattr(self, 'tdb_reparam'):
            # 推理模式：单卷积+激活
            return self.nonlinear(self.tdb_reparam(inputs))
        else:
            # 训练模式：三个分支相加+激活
            out1 = self.duplicate1(inputs)
            out2 = self.duplicate2(inputs)
            out3 = self.self_bn(self.identity_conv(inputs))
            # 调试打印shape（可选）
            # print(f"分支1shape: {out1.shape}, 分支2shape: {out2.shape}, 分支3shape: {out3.shape}")
            return self.nonlinear(out1 + out2 + out3)



class AMBB_3_1_3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 deploy=False, nonlinear=None):
        super(AMBB_3_1_3, self).__init__()
        self.deploy = deploy
        self.nonlinear = nonlinear if nonlinear is not None else nn.Identity()
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.groups = groups
        self.dilation = dilation
        self.stride = stride
        self.in_channels = in_channels
        self.padding = padding
        assert padding == kernel_size // 2, "Padding must be kernel_size//2 for size matching"

        if deploy:
            # 部署模式：单K×K卷积
            self.tdb_reparam = nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True
            )
        else:
            # 分支1：1×1 Conv + BN（点卷积分支）
            self.duplicate1 = conv_bn(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                stride=stride, padding=0, dilation=dilation, groups=groups
            )

            # 分支2：1×1 Conv+BN → K×K Conv+BN（串联结构）
            self.duplicate2_1x1 = conv_bn(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                stride=stride, padding=0, dilation=dilation, groups=groups
            )
            self.duplicate2 = conv_bn(
                in_channels=out_channels,  # 输入通道=1×1的输出通道
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,  # 注意：此处stride=1，因为前面的1×1已做stride
                padding=padding, dilation=dilation, groups=groups
            )

            # 分支3：K×K Conv + BN（辅助空间分支）
            self.duplicate3 = conv_bn(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation, groups=groups
            )

            # 非对称卷积分支：1×1 Conv+BN → K×1 Conv（垂直分支）
            self.ver_1x1 = conv_bn(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                stride=stride, padding=0, dilation=dilation, groups=groups
            )
            ver_padding = (kernel_size // 2, 0)  # K×1卷积的padding
            self.ver_conv = nn.Conv2d(
                in_channels=out_channels,  # 输入通道=1×1的输出通道（关键修正）
                out_channels=out_channels,
                kernel_size=(kernel_size, 1),
                stride=1,  # stride由前面的1×1卷积完成
                padding=ver_padding, dilation=dilation, groups=groups,
                bias=False, padding_mode="zeros"
            )
            self.ver_bn = nn.BatchNorm2d(num_features=out_channels, affine=True)

            # 非对称卷积分支：1×1 Conv+BN → 1×K Conv（水平分支）
            self.hor_1x1 = conv_bn(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                stride=stride, padding=0, dilation=dilation, groups=groups
            )
            hor_padding = (0, kernel_size // 2)  # 1×K卷积的padding
            self.hor_conv = nn.Conv2d(
                in_channels=out_channels,  # 输入通道=1×1的输出通道（关键修正）
                out_channels=out_channels,
                kernel_size=(1, kernel_size),
                stride=1,  # stride由前面的1×1卷积完成
                padding=hor_padding, dilation=dilation, groups=groups,
                bias=False, padding_mode="zeros"
            )
            self.hor_bn = nn.BatchNorm2d(num_features=out_channels, affine=True)

            # 裁剪标记（典型场景下为0）
            self.crop = 0

    def _fuse_bn_tensor(self, conv, bn):
        """融合单个Conv+BN的核和偏置"""
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return conv.weight * t, bn.bias - bn.running_mean * bn.weight / std

    def _add_to_square_kernel(self, square_kernel, asym_kernel):
        """将非方形核叠加到方形核的中心（非in-place操作）"""
        square_kernel = square_kernel.clone()
        asym_h, asym_w = asym_kernel.size(2), asym_kernel.size(3)
        square_h, square_w = square_kernel.size(2), square_kernel.size(3)
        h_start = square_h // 2 - asym_h // 2
        w_start = square_w // 2 - asym_w // 2
        square_kernel[:, :, h_start:h_start + asym_h, w_start:w_start + asym_w] += asym_kernel
        return square_kernel

    def get_equivalent_kernel_bias(self):
        """融合所有分支为单个K×K卷积的核和偏置（重点修复非对称分支的串联融合）"""
        device = next(self.parameters()).device
        K = self.kernel_size

        # --------------------------- 分支2：1×1 → K×K 串联融合 ---------------------------
        # 融合1×1 Conv+BN
        k2_1x1, b2_1x1 = self._fuse_bn_tensor(self.duplicate2_1x1.conv, self.duplicate2_1x1.bn)
        # 融合K×K Conv+BN
        k2_kxk, b2_kxk = self._fuse_bn_tensor(self.duplicate2.conv, self.duplicate2.bn)
        # 串联融合
        dup2_k, dup2_b = transIII_1x1_kxk(k2_1x1, b2_1x1, k2_kxk, b2_kxk, self.groups)

        # --------------------------- 分支1：1×1 → 扩展为K×K核 ---------------------------
        dup1_k1x1, dup1_b = self._fuse_bn_tensor(self.duplicate1.conv, self.duplicate1.bn)
        dup1_k = torch.zeros_like(dup2_k, device=device)
        dup1_k = self._add_to_square_kernel(dup1_k, dup1_k1x1)

        # --------------------------- 分支3：K×K 直接融合 ---------------------------
        dup3_k, dup3_b = self._fuse_bn_tensor(self.duplicate3.conv, self.duplicate3.bn)

        # --------------------------- 垂直分支：1×1 → K×1 串联融合 ---------------------------
        # 融合1×1 Conv+BN
        ver_1x1_k, ver_1x1_b = self._fuse_bn_tensor(self.ver_1x1.conv, self.ver_1x1.bn)
        # 融合K×1 Conv+BN
        ver_conv_k, ver_conv_b = transI_fusebn(self.ver_conv.weight, self.ver_bn)
        # 串联融合
        ver_k, ver_b = transIII_1x1_kxk(ver_1x1_k, ver_1x1_b, ver_conv_k, ver_conv_b, self.groups)
        # 将K×1核pad为K×K
        pad_w = (K - ver_k.size(3)) // 2
        ver_k_pad = F.pad(ver_k, (pad_w, pad_w, 0, 0))

        # --------------------------- 水平分支：1×1 → 1×K 串联融合 ---------------------------
        # 融合1×1 Conv+BN
        hor_1x1_k, hor_1x1_b = self._fuse_bn_tensor(self.hor_1x1.conv, self.hor_1x1.bn)
        # 融合1×K Conv+BN
        hor_conv_k, hor_conv_b = transI_fusebn(self.hor_conv.weight, self.hor_bn)
        # 串联融合
        hor_k, hor_b = transIII_1x1_kxk(hor_1x1_k, hor_1x1_b, hor_conv_k, hor_conv_b, self.groups)
        # 将1×K核pad为K×K
        pad_h = (K - hor_k.size(2)) // 2
        hor_k_pad = F.pad(hor_k, (0, 0, pad_h, pad_h))

        # --------------------------- 所有分支核和偏置相加 ---------------------------
        k_total = dup1_k + dup2_k + dup3_k + ver_k_pad + hor_k_pad
        b_total = dup1_b + dup2_b + dup3_b + ver_b + hor_b

        return k_total, b_total

    def switch_to_deploy(self):
        """切换到部署模式，删除所有训练分支"""
        deploy_k, deploy_b = self.get_equivalent_kernel_bias()
        self.deploy = True

        # 创建部署用的K×K卷积
        self.tdb_reparam = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=True
        )

        # 赋值融合后的核和偏置（禁用梯度）
        with torch.no_grad():
            self.tdb_reparam.weight.copy_(deploy_k)
            self.tdb_reparam.bias.copy_(deploy_b)

        # 删除所有训练分支（包含ver_1x1/hor_1x1）
        del_attrs = [
            'duplicate1', 'duplicate2_1x1', 'duplicate2', 'duplicate3',
            'ver_1x1', 'ver_conv', 'ver_bn', 'hor_1x1', 'hor_conv', 'hor_bn'
        ]
        for attr in del_attrs:
            if hasattr(self, attr):
                self.__delattr__(attr)

    def forward(self, inputs):
        if self.deploy and hasattr(self, 'tdb_reparam'):
            return self.nonlinear(self.tdb_reparam(inputs))

        # 训练模式：各分支前向计算
        out1 = self.duplicate1(inputs)
        out2 = self.duplicate2(self.duplicate2_1x1(inputs))
        out3 = self.duplicate3(inputs)

        # 垂直分支：1×1 → K×1 → BN
        ver_out = self.ver_bn(self.ver_conv(self.ver_1x1(inputs)))
        # 水平分支：1×1 → 1×K → BN
        hor_out = self.hor_bn(self.hor_conv(self.hor_1x1(inputs)))

        # 裁剪（典型场景下为0）
        if self.crop > 0:
            ver_out = ver_out[:, :, self.crop:-self.crop, :]
            hor_out = hor_out[:, :, :, self.crop:-self.crop]

        # 总输出
        total_out = out1 + out2 + out3 + ver_out + hor_out
        return self.nonlinear(total_out)

if __name__ == '__main__':
    N = 1
    C = 64
    H = 128
    W = 64
    O = 32
    groups = 4

    x = torch.randn(N, C, H, W)
    print('input shape is ', x.size())

    test_kernel_padding = [(3, 1)]
    for k, p in test_kernel_padding:
        ambb = AMBB(64, 128, kernel_size=k, padding=p, stride=2, deploy=False)
        ambb.eval()
        for module in ambb.modules():
            if isinstance(module, nn.BatchNorm2d):
                nn.init.uniform_(module.running_mean, 0, 0.1)
                nn.init.uniform_(module.running_var, 0, 0.2)
                nn.init.uniform_(module.weight, 0, 0.3)
                nn.init.uniform_(module.bias, 0, 0.4)
        out = ambb(x)
        ambb.switch_to_deploy()
        deployout = ambb(x)
        print('difference between the outputs of the training-time and converted ambb is')
        print(((deployout - out) ** 2).sum())
