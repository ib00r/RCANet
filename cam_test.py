import numpy as np
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image

import pytorch_grad_cam
from pytorch_grad_cam.utils.image import show_cam_on_image
from unet import Unet



# model = models.resnet18(pretrained = True)
model = Unet()
num_classes = model.num_classes
# model = Unet().net.module
model = model.net.module
model.train()



# target_layers = [model.layer4[1].bn2]
target_layers = [model.conv_encode4[0][4]]

origin_img = cv2.imread('./img/A058_0_256_1792.jpg')
# origin_img = cv2.imread('./img/Snipaste_2024-06-04_20-59-23.png')
rgb_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)
rgb_img = Image.fromarray(rgb_img)

trans = transforms.Compose({
    transforms.ToTensor(),
    transforms.Resize(224),
    transforms.CenterCrop(224)
})

crop_img = trans(rgb_img)
net_input = transforms.Normalize((0.485, 0.456, 0.406), (0.228, 0.224, 0.225))(crop_img).unsqueeze(0)

net_input = net_input.cuda()
net_input.requires_grad_(True)


canvas_img = (crop_img * 255).byte().numpy().transpose(1, 2, 0)
canvas_img = cv2.cvtColor(canvas_img, cv2.COLOR_RGB2BGR)

# net_input = net_input.cuda()
# output = model(net_input)
# print(type(output))

# cam = pytorch_grad_cam.GradCAMPlusPlus(model = model, target_layers = target_layers)
cam = pytorch_grad_cam.GradCAM(model = model, target_layers = target_layers)


target_category = 0
fake_target = torch.zeros([1, 224, 224], dtype=torch.float).cuda()#.scatter_(1, torch.tensor([target_category]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1), 1.0).cuda()
fake_target[:, 112, 112] = target_category
# fake_target = torch.zeros([1, num_classes, 1, 1]).scatter_(1, torch.tensor([[target_category]]), 1.0)

# 执行模型的正向传播
with torch.no_grad():
    output = model(net_input)
    # 定义损失函数，这里使用交叉熵损失作为示例
selected_class_output = output[:, 1, :, :]
loss = F.binary_cross_entropy_with_logits(selected_class_output, fake_target)

# 执行反向传播
loss.backward(retain_graph=False)


grayscale_cam = cam(net_input)
grayscale_cam = grayscale_cam[0, :]

src_img = np.float32(canvas_img) / 255
visualization_img = show_cam_on_image(src_img, grayscale_cam, use_rgb=False)
cv2.imshow('feature_map', visualization_img)
cv2.waitKey(0)