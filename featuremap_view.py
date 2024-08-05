import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
from unet import Unet
import imageio

# 导入数据
def get_image_info(image_dir):
    # 以RGB格式打开图像
    # Pytorch DataLoader就是使用PIL所读取的图像格式
    # 建议就用这种方法读取图像，当读入灰度图像时convert('')
    image_info = Image.open(image_dir).convert('RGB')
    # 数据预处理方法
    image_transform = transforms.Compose([
        transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_info = image_transform(image_info)
    image_info = image_info.unsqueeze(0)
    return image_info

# 获取第k层的特征图
def get_k_layer_feature_map(feature_extractor, k, x):
    with torch.no_grad():
        index = 0
        for layer in feature_extractor:
            x = layer(x)
            if k == index:
                return x
            index += 1

#  可视化特征图
def show_feature_map(feature_map):
    feature_map = feature_map.squeeze(0)
    feature_map = feature_map.cpu().numpy()
    feature_map_num = feature_map.shape[0]
    row_num = np.ceil(np.sqrt(feature_map_num))
    plt.figure()
    for index in range(1, feature_map_num+1):
        plt.subplot(int(row_num), int(row_num), index)
        plt.imshow(feature_map[index-1], cmap='jet')
        plt.axis('off')
        # scipy.misc.imsave(str(index)+".png", feature_map[index-1])
        image_array_int = np.clip(feature_map[index-1] * 255, 0, 255).astype(np.uint8)
        imageio.imwrite("./feature_out/"+str(index)+".png", image_array_int)
    plt.show()




if __name__ ==  '__main__':
    # 初始化图像的路径
    image_dir = r"./img/A499_0_1536_1792.jpg"
    # 定义提取第几层的feature map
    k = 4
    # 导入Pytorch封装的AlexNet网络模型

    # 是否使用gpu运算
    # use_gpu = torch.cuda.is_available()
    # use_gpu =False
    # 读取图像信息
    image_info = get_image_info(image_dir).cuda()
    model = Unet()
    # 判断是否使用gpu
    # if use_gpu:
    # if True:
        # model = model.detect_image(image_info).cuda()
        # image_info = image_info.cuda()
    # alexnet只有features部分有特征图
    # classifier部分的feature map是向量
    feature_extractor = model.net.module
    # feature_map = get_k_layer_feature_map(feature_extractor, k, image_info)
    feature_map = feature_extractor(image_info)
    feature_map_cpy = feature_map.detach()
    show_feature_map(feature_map_cpy)