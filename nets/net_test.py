# import vgg
# import torch
# X = torch.randn(2,3,256,256)
# model = vgg.VGG16(False,3)
# for name,layer in model.named_modules():
#     X = layer(X)
#     print(name, 'output shape:\t',X.shape)
#     # print(layer.__class__.__name__,X.shape())


import vgg
import torch
import resnet
X = torch.randn(2, 3, 256, 256)
# model = vgg.VGG16(False, 3)
model = resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3])
i = 0
for name, layer in model.named_modules():
    if i <= 2:
        print(name)
        print(layer)
        i += 1
        print("#-----------------------------------------------#")
        continue
    X = layer(X)
    # print(len(X))
    # if isinstance(X, list):
    #     for i, x in enumerate(X):
    #         print(name, f'output {i} shape:\t', x.shape)
    # else:
    #     print(name, 'output shape:\t', X.shape)
    print(name, f'output shape:\t', X.shape)
    # i += 1
    break
    # print(name)