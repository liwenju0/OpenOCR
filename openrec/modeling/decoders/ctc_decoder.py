import torch.nn as nn
import torch.nn.functional as F


class CTCDecoder(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels=6625,
                 mid_channels=None,
                 return_feats=False,
                 **kwargs):
        super(CTCDecoder, self).__init__()
        if mid_channels is None:
            self.fc = nn.Linear(
                in_channels,
                out_channels,
                bias=True,
            )
        else:
            self.fc1 = nn.Linear(
                in_channels,
                mid_channels,
                bias=True,
            )
            self.fc2 = nn.Linear(
                mid_channels,
                out_channels,
                bias=True,
            )

        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.return_feats = return_feats

    def forward(self, x, data=None):
        if self.mid_channels is None:
            predicts = self.fc(x)
        else:
            x = self.fc1(x)
            predicts = self.fc2(x)

        if self.return_feats:
            result = (x, predicts)
        else:
            result = predicts

        if not self.training:
            # 在推理阶段不使用softmax
            # 原因:
            # 1. CTC解码通常使用log_softmax而不是softmax
            # 2. 许多后处理步骤(如beam search)直接使用logits更有效
            # 3. 避免不必要的计算,提高推理速度
            # predicts = F.softmax(predicts, dim=2)
            result = predicts

        return result
