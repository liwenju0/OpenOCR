import torch
from torch import nn


class EmbeddingLoss(nn.Module):
    def __init__(self, cfg=None, **kwargs):
        super(EmbeddingLoss, self).__init__()
        self.cfg  = cfg
        self.load_dict()
        self.char_num = len(self.dict)

    def load_dict(self):
        if self.cfg is not None:
            self.dict_path = self.cfg['Global']['character_dict_path']
        else:
            self.dict_path = "/home/deepctrl/liwenju/ocr-train/all_vocab.txt"

        self.dict = {0: "blank"}
        with open(self.dict_path, 'r', encoding='utf-8') as f:
            for index, line in enumerate(f):
                char = line.strip().strip("\r\n")
                self.dict[int(index)+1] = char
            self.dict[index+2] = " "

    def forward(self, embeddings: torch.nn.Linear):
        weight = embeddings.weight.T.transpose(0, 1)
        
        # 检查权重是否包含 nan 或 inf
        if torch.isnan(weight).any() or torch.isinf(weight).any():
            print("警告：权重中包含 nan 或 inf 值")
            return {'EMB/loss': torch.tensor(0.0), 'EMB/mean_distance': torch.tensor(0.0)}

        # 计算欧几里得距离矩阵
        distances = torch.cdist(weight, weight, p=2)
        
        # 将对角线元素设置为一个大数，而不是无穷大
        distances = distances + torch.eye(self.char_num, device=distances.device) * 1e6
        
        # 对每个特征,找到最近的距离
        min_distances, min_indices = torch.min(distances, dim=1)
        

        # # 在GPU上创建张量
        # similar_char = torch.zeros((self.char_num, 2), dtype=torch.long, device=min_distances.device)
        # similar_char[:, 0] = torch.arange(self.char_num, device=min_distances.device)
        # similar_char[:, 1] = min_indices
        # # 在GPU上进行排序
        # _, sorted_indices = torch.sort(min_distances)
        # similar_char = similar_char[sorted_indices[:50]]
        # similar_char = {f"EMB/[{self.dict[i.item()]} <-> {self.dict[j.item()]}]": round(min_distances[i].item(), 3)
        #                  for i, j in similar_char.cpu().numpy()}
        # 计算min_distances的平均值
        mean_distance = min_distances.mean()
        
        # 将大于平均值的距离置为0
        min_distances = torch.where(min_distances > mean_distance, torch.zeros_like(min_distances), min_distances)
        
        # 计算平均最小距离作为损失
        loss = -min_distances.mean()
        
        # 在返回之前检查 loss 和 mean_distance
        ret = {'EMB/loss': loss, 'EMB/mean_distance': mean_distance}
        # ret.update(similar_char)
        return ret

class CTCLoss(nn.Module):

    def __init__(self, use_focal_loss=False, zero_infinity=False, **kwargs):
        super(CTCLoss, self).__init__()
        self.loss_func = nn.CTCLoss(blank=0,
                                    reduction='none',
                                    zero_infinity=zero_infinity)
        self.use_focal_loss = use_focal_loss

    def forward(self, predicts, batch):
        # predicts = predicts['res']

        batch_size = predicts.size(0)
        label, label_length = batch[1], batch[2]
        predicts = predicts.log_softmax(2)
        predicts = predicts.permute(1, 0, 2)
        preds_lengths = torch.tensor([predicts.size(0)] * batch_size,
                                     dtype=torch.long)
        loss = self.loss_func(predicts, label, preds_lengths, label_length)

        if self.use_focal_loss:
            # 使用 torch.clamp 限制 loss 的范围，避免指数计算时溢出
            clamped_loss = torch.clamp(loss, min=-20, max=20)
            weight = 1 - torch.exp(-clamped_loss)
            weight = torch.square(weight)
            # 使用 torch.where 避免乘以零权重
            loss = torch.where(weight > 0, loss * weight, loss)
        loss = loss.mean()
        return {'loss': loss}
