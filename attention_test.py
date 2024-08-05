from fightingcv_attention.attention.SEAttention import SEAttention
import torch

input=torch.randn(50,512,7,7)
se = SEAttention(channel=512,reduction=16)
output=se(input)
print(output.shape)
