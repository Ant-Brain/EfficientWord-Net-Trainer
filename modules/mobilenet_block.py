import torch.nn as nn
from modules.attention import SelfAttentionConv2d, SelfAttentionConv2dLite

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, _make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(_make_divisible(channel // reduction, 8), channel),
                h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )

def conv_2x2_bn(inp, oup, stride, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 2, stride, padding, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )

def give_se_layer(channels, use_se, use_attention, lite_attention) :
    if use_se :
        return SELayer(channels)
    else:
        if use_attention :
            return SelfAttentionConv2dLite(channels) if lite_attention else SelfAttentionConv2d(channels)
    return nn.Identity()

class InvertedResidual2D(nn.Module):
    def __init__(self, inp, hidden_ratio, out_channel, kernel_size, stride, interim_layer:nn.Module, use_hs:bool):
        hidden_dim = int(inp*hidden_ratio)
        super(InvertedResidual2D, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == out_channel
        #print(interim_layer)
        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                interim_layer(hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, out_channel, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channel),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                interim_layer(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, out_channel, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channel),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)
