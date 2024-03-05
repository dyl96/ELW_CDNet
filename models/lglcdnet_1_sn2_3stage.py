from . import shufflenetv2
import warnings
import torch
from torch import Tensor
from typing import Optional, Tuple
from torch.nn import functional as F
import torch.nn as nn


class Dropout(nn.Dropout):
    """
    This layer, during training, randomly zeroes some of the elements of the input tensor with probability `p`
    using samples from a Bernoulli distribution.

    Args:
        p: probability of an element to be zeroed. Default: 0.5
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where :math:`N` is the batch size
        - Output: same as the input

    """

    def __init__(
        self, p: Optional[float] = 0.5, inplace: Optional[bool] = False, *args, **kwargs
    ) -> None:
        super().__init__(p=p, inplace=inplace)

    def profile_module(
        self, input: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, float, float]:
        return input, 0.0, 0.0


class LinearSelfAttention(nn.Module):
    """
    This layer applies a self-attention with linear complexity, as described in `MobileViTv2 <https://arxiv.org/abs/2206.02680>`_ paper.
    This layer can be used for self- as well as cross-attention.

    Args:
        opts: command line arguments
        embed_dim (int): :math:`C` from an expected input of size :math:`(N, C, H, W)`
        attn_dropout (Optional[float]): Dropout value for context scores. Default: 0.0
        bias (Optional[bool]): Use bias in learnable layers. Default: True

    Shape:
        - Input: :math:`(N, C, P, N)` where :math:`N` is the batch size, :math:`C` is the input channels,
        :math:`P` is the number of pixels in the patch, and :math:`N` is the number of patches
        - Output: same as the input

    .. note::
        For MobileViTv2, we unfold the feature map [B, C, H, W] into [B, C, P, N] where P is the number of pixels
        in a patch and N is the number of patches. Because channel is the first dimension in this unfolded tensor,
        we use point-wise convolution (instead of a linear layer). This avoids a transpose operation (which may be
        expensive on resource-constrained devices) that may be required to convert the unfolded tensor from
        channel-first to channel-last format in case of a linear layer.
    """

    def __init__(
        self,
        embed_dim: int,
        attn_dropout: Optional[float] = 0.0,
        bias: Optional[bool] = True,
    ) -> None:
        super(LinearSelfAttention, self).__init__()

        self.qkv_proj = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=1 + (2 * embed_dim),
            bias=bias,
            kernel_size=1
        )
        self.attn_dropout = Dropout(p=attn_dropout)
        self.out_proj = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            bias=bias,
            kernel_size=1,
        )
        self.embed_dim = embed_dim

    def _forward_self_attn(self, x: Tensor) -> Tensor:
        # [B, C, P, N] --> [B, h + 2d, P, N]
        qkv = self.qkv_proj(x)

        # Project x into query, key and value
        # Query --> [B, 1, P, N]
        # value, key --> [B, d, P, N]
        query, key, value = torch.split(
            qkv, split_size_or_sections=[1, self.embed_dim, self.embed_dim], dim=1
        )

        # apply softmax along N dimension
        context_scores = F.softmax(query, dim=-1)

        # Compute context vector
        # [B, d, P, N] x [B, 1, P, N] -> [B, d, P, N]
        context_vector = key * context_scores
        # [B, d, P, N] --> [B, d, P, 1]
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        # combine context vector with values
        # [B, d, P, N] * [B, d, P, 1] --> [B, d, P, N]
        out = F.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        return out

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_self_attn(x)


class LightGlobalLocalFEM(nn.Module):
    def __init__(self, in_d, out_d):
        super(LightGlobalLocalFEM, self).__init__()
        self.skip = nn.Conv2d(in_d, out_d, 1)
        self.local_rep = nn.Sequential(nn.Conv2d(in_d, out_d, 1),
                                       nn.Conv2d(out_d, out_d, 3, 1, 1),
                                       nn.Conv2d(out_d, out_d, 3, 1, 1))
        self.global_rep = nn.Sequential(nn.Conv2d(in_d, out_d, 1),
                                        LinearSelfAttention(out_d),
                                        LinearSelfAttention(out_d))

    def forward(self, x):
        out = self.skip(x) + self.local_rep(x) + self.global_rep(x)
        return out


class FEM(nn.Module):
    def __init__(self, in_d=(), out_d=32):
        super(FEM, self).__init__()
        self.fem1 = LightGlobalLocalFEM(in_d=in_d[0], out_d=out_d)
        self.fem2 = LightGlobalLocalFEM(in_d=in_d[1], out_d=out_d)
        self.fem3 = LightGlobalLocalFEM(in_d=in_d[2], out_d=out_d)

    def forward(self, inputs):
        feat1 = self.fem1(inputs[0])
        feat2 = self.fem2(inputs[1])
        feat3 = self.fem3(inputs[2])

        return feat1, feat2, feat3


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > input_w:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class TemporalFeatureFusionModule(nn.Module):
    def __init__(self, in_d, out_d):
        super(TemporalFeatureFusionModule, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.relu = nn.ReLU(inplace=True)
        # branch 1
        self.conv_branch1 = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=7, dilation=7),
            nn.BatchNorm2d(self.in_d)
        )
        # branch 2
        self.conv_branch2 = nn.Conv2d(self.in_d, self.in_d, kernel_size=1)
        self.conv_branch2_f = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.BatchNorm2d(self.in_d)
        )
        # branch 3
        self.conv_branch3 = nn.Conv2d(self.in_d, self.in_d, kernel_size=1)
        self.conv_branch3_f = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.BatchNorm2d(self.in_d)
        )
        # branch 4
        self.conv_branch4 = nn.Conv2d(self.in_d, self.in_d, kernel_size=1)
        self.conv_branch4_f = nn.Sequential(
            nn.Conv2d(self.in_d, self.out_d, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.out_d)
        )
        self.conv_branch5 = nn.Conv2d(self.in_d, self.out_d, kernel_size=1)

    def forward(self, x1, x2):
        # temporal fusion
        x = torch.abs(x1 - x2)
        # branch 1
        x_branch1 = self.conv_branch1(x)
        # branch 2
        x_branch2 = self.relu(self.conv_branch2(x) + x_branch1)
        x_branch2 = self.conv_branch2_f(x_branch2)
        # branch 3
        x_branch3 = self.relu(self.conv_branch3(x) + x_branch2)
        x_branch3 = self.conv_branch3_f(x_branch3)
        # branch 4
        x_branch4 = self.relu(self.conv_branch4(x) + x_branch3)
        x_branch4 = self.conv_branch4_f(x_branch4)
        x_out = self.relu(self.conv_branch5(x) + x_branch4)

        return x_out


class ProgressiveTemporalFeatureFusionModule(nn.Module):
    def __init__(self, in_d):
        super(ProgressiveTemporalFeatureFusionModule, self).__init__()
        self.in_d = in_d
        self.relu = nn.ReLU(inplace=True)
        self.conv_s3 = nn.Sequential(
            nn.Conv2d(3*self.in_d, 3*self.in_d, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.BatchNorm2d(3*self.in_d),
            nn.ReLU(inplace=True)
        )
        self.conv_s4 = nn.Sequential(
            nn.Conv2d(2*self.in_d, 2*self.in_d, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.BatchNorm2d(2*self.in_d)
        )

        self.conv_s5 = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.in_d)
        )

    def forward(self, x3, x4, x5):
        x = resize(self.conv_s5(x5), scale_factor=(2, 2), mode='bilinear')   # 16x
        x = resize(self.conv_s4(torch.cat((x, x4), dim=1)), scale_factor=(2, 2), mode='bilinear')   # 8x
        x = resize(self.conv_s3(torch.cat((x, x3), dim=1)), scale_factor=(2, 2), mode='bilinear')  # 4x

        return x


class Decoder(nn.Module):
    def __init__(self, in_d=32):
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_d, int(in_d/2), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(in_d/2)),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(int(in_d/2), int(in_d/4), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(in_d/4)),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(int(in_d/4), int(in_d/8), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(in_d/8)),
        )
        self.cls = nn.Conv2d(int(in_d/8), 1, kernel_size=1)

    def forward(self, x):
        # high-level
        x = self.conv1(x)
        x = resize(x, scale_factor=(2, 2), mode='bilinear')
        x = self.conv2(x)
        x = resize(x, scale_factor=(2, 2), mode='bilinear')
        x = self.conv3(x)
        x = self.cls(x)

        mask = torch.sigmoid(x)

        return mask


class BaseNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, model_size='0.5x', fe_nc=32):
        super(BaseNet, self).__init__()
        self.backbone = shufflenetv2.ShuffleNetV2(model_size=model_size, pretrain=True)

        if model_size == '0.5x':
            channels = [24, 24, 48, 96, 192]
        else:
            channels = [24, 24, 116, 232, 464]

        # self.backbone = MobileNetV2.mobilenet_v2()
        # channels = [16, 24, 32, 96, 320]

        self.fem1 = FEM(in_d=channels[2:], out_d=fe_nc)
        self.fem2 = FEM(in_d=channels[2:], out_d=fe_nc)

        self.ptffm = ProgressiveTemporalFeatureFusionModule(in_d=fe_nc)
        self.decoder = Decoder(3 * fe_nc)

    def forward(self, x1, x2):
        # forward backbone resnet
        x1_1, x1_2, x1_3, x1_4, x1_5 = self.backbone(x1)
        x2_1, x2_2, x2_3, x2_4, x2_5 = self.backbone(x2)
        # feature enhancement
        x1_3, x1_4, x1_5 = self.fem1((x1_3, x1_4, x1_5))
        x2_3, x2_4, x2_5 = self.fem2((x2_3, x2_4, x2_5))
        # feature fusion
        x3, x4, x5 = torch.abs(x1_3 - x2_3), torch.abs(x1_4 - x2_4), torch.abs(x1_5 - x2_5)
        # temporal fusion
        diff_feats = self.ptffm(x3, x4, x5)
        #
        mask = self.decoder(diff_feats)
        return mask


if __name__ == '__main__':
    model = BaseNet()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
