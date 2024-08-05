import cv2
from PIL import Image

img_path = "./VOCdevkit/SegmentationClass/A033_1_2048_2048.png"
#img = cv2.imread(img_path)
#cv2.imshow("previous", img)
img = Image.open(img_path)
img.show()
#cv2.imshow("将mask颜色改为白色", img * 255)

