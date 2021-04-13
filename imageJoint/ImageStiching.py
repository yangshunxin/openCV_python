from imageJoint.Stitcher import Stitcher
import cv2
from cvBaseOperation import cv_show

# 读取拼接图片
imageA = cv2.imread("./images/left_01.png")
imageB = cv2.imread("./images/right_01.png")
cv_show(imageA, "Image A")
cv_show(imageB, "Image B")

# 把图片拼接成全景图
stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)

# 显示所有图片

cv_show(vis, "Keypoint Matches")
cv_show(result, "Result")