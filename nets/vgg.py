import torch.nn as nn
from torch.hub import load_state_dict_from_url
from fightingcv_attention.attention.SEAttention import SEAttention
from fightingcv_attention.attention.CBAM import CBAMBlock


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        #----------------------------------------------
        #调整注意力位置时，channel要调整
        #----------------------------------------------

        # self.se = SEAttention(channel=64, reduction=16)
        # self.cbam = CBAMBlock(channel=64, reduction=16, kernel_size=49)
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()


    def forward(self, x):
        # x = self.features(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)
        ###############################################################
        #   前向传播过程，每次保留原来的输入  到最后跨越连接 特征融合
        ###############################################################

        ###############################################################
        #  添加注意力
        ###############################################################

#-----------------------------------------------------------
# 这是没有用批量归一化层 对应的features
#-----------------------------------------------------------

        feat1 = self.features[  :4 ](x)#表示将输入数据 x 传递给 features 中的第 0 到 3 个卷积层
        # feat1 = self.cbam(feat1)#加在最初的卷积层后
        # feat1 = self.se(feat1)#加在最初的卷积层后
        feat2 = self.features[4 :9 ](feat1)
        feat3 = self.features[9 :16](feat2)
        feat4 = self.features[16:23](feat3)
        # feat4 = self.se(feat4)  # 加在后面的卷积层
        feat5 = self.features[23:-1](feat4)

        # -----------------------------------------------------------
        # 这是使用批量归一化层 对应的features
        # -----------------------------------------------------------
        #
        # feat1 = self.features[  :6 ](x)#表示将输入数据 x 传递给 features 中的第 0 到 3 个卷积层
        # # feat1 = self.cbam(feat1)#加在最初的卷积层后
        # feat1 = self.se(feat1)#加在最初的卷积层后
        # feat2 = self.features[6 :13 ](feat1)
        # feat3 = self.features[13 :23](feat2)
        # feat4 = self.features[23:33](feat3)
        # # feat4 = self.se(feat4)  # 加在后面的卷积层
        # feat5 = self.features[33:-1](feat4)
        #
        return [feat1, feat2, feat3, feat4, feat5]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                ###############################################################
                #   卷积层的初始化 用了kaiming初始化
                ###############################################################
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            ###############################################################
            #   对于批量归一化层
            ###############################################################
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                #从SEAttention抄过来的写法
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False, in_channels = 3):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
# 512,512,3 -> 512,512,64 -> 256,256,64 -> 256,256,128 -> 128,128,128 -> 128,128,256 -> 64,64,256
# 64,64,512 -> 32,32,512 -> 32,32,512
cfgs = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}

#--------------------------------------------------------------------
# 在这里修改是否使用BatchNorm层   如果使用会在每个卷积层后加入BatchNorm层
# 注意，还需要更改vgg16 的forward方法，匹配对应的网络架构
#--------------------------------------------------------------------

def VGG16(pretrained, in_channels = 3, **kwargs):
    model = VGG(make_layers(cfgs["D"], batch_norm = False, in_channels = in_channels), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth", model_dir="./model_data")
        model.load_state_dict(state_dict)
    
    del model.avgpool
    del model.classifier
    return model

# if __name__ == '__main__':
#     vgg = VGG16(False)