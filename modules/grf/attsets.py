import torch
import torch.nn as nn


def dense(dim_in: int, dim_out: int):
    return nn.Sequential(
        nn.Linear(dim_in, dim_out),
        nn.ReLU(True)
    )


class attsets(nn.Module):
    def __init__(self, dim_feat: int, dim_pts: int, attn_out_len: int = 512):
        super(attsets, self).__init__()

        self.dense1_3 = nn.Sequential(
            dense(dim_feat + dim_pts, 256),
            dense(256, 256),
            dense(256, 256)
        )
        self.dense4 = dense(256 + dim_feat + dim_pts, attn_out_len)

        self.dense5 = dense(attn_out_len, attn_out_len)

        self.dense6 = dense(attn_out_len, attn_out_len)

    def forward(self, inputs, embedded_pts):
        """
        process
        :param inputs: (b, n, i, p, di)
        :param embedded_pts: (b, n, i, p, dp)
        :return:
        """
        inputs_init = torch.cat([inputs, embedded_pts], dim=-1)

        inputs = self.dense1_3(inputs_init)
        inputs = torch.cat([inputs, inputs_init], dim=-1)
        inputs = self.dense4(inputs)

        mask = self.dense5(inputs)
        mask = torch.softmax(mask, dim=1)
        att = inputs * mask
        output = torch.sum(att, dim=1)  # (b, i, p, attn_out_len)

        output = self.dense6(output)

        return output
