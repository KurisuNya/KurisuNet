from itertools import pairwise

import torch.nn as nn


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class CNNClassifier(nn.Module):
    def __init__(self, in_ch: int, class_num: int, width: float = 1.0):
        super(CNNClassifier, self).__init__()

        channels = [16, 32, 64, 128, 256, 512]
        channels = [_make_divisible(ch * width, 4) for ch in channels]
        channels.insert(0, in_ch)
        channel_pairs = pairwise(channels)

        self.__conv_seq = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
                for in_ch, out_ch in channel_pairs
            ]
        )
        self.__pool = nn.AdaptiveAvgPool2d((1, 1))
        self.__conv_head = nn.Conv2d(channels[-1], channels[-1], 1, 1, 0, bias=True)
        self.__act = nn.ReLU(inplace=True)
        self.__fc = nn.Linear(channels[-1], class_num)

    def forward(self, x):
        x = self.__conv_seq(x)
        x = self.__pool(x)
        x = self.__conv_head(x)
        x = self.__act(x)
        x = x.view(x.size(0), -1)
        x = self.__fc(x)
        return x


if __name__ == "__main__":
    from kurisuinfo import summary

    model = CNNClassifier(in_ch=1, class_num=2, width=0.5)
    summary(model, (1, 1, 128, 128))
