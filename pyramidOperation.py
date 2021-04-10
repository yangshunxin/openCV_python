import cv2
import numpy as np
import matplotlib.pyplot as plt
from cvBaseOperation import cv_show

def gaosiPyramidTest():
    """
        图像金字塔分为：高斯金字塔和拉普拉斯金字塔
        图像金字塔表示：同一张图像，按照不同比例进行缩放不同的大小，组合在一起就是图像金字塔
        图像金字塔的操作分为两种：向下采样（缩小）和向上采样（放大）
        高斯金字塔：向下采样方法（缩小）
            Gkernel = 1/16 *[   [1, 4, 6, 4, 1],
                                [4, 16, 24, 16, 4],
                                [6, 24, 36, 24, 6],
                                [4, 16, 24, 16, 4],
                                [1, 4, 6, 4, 1]]
            1.将G与高斯内核卷积，2.将所有偶数行和列去除


        高斯金字塔：向上采样方法（放大）
            [[10, 30],[56, 96]]     ->  [[10, 0, 30, 0],[0, 0, 0, 0],[56, 0, 96, 0], [0, 0, 0, 0]]
            1.将图像在每个方向扩大为原来的两倍，新增的行和列以0填充
            2.使用先前同样的内核（乘以4）与放大后的图像卷积，获得近似值
    """
    img = cv2.imread("./images/AM.png")
    # cv_show(img)
    print("origin shape:{}".format(img.shape))
    up = cv2.pyrUp(img)
    print("up shape:{}".format(up.shape))
    # cv_show(up)
    down = cv2.pyrDown(img)  # 下采样损失很多信息
    print("down shape:{}".format(down.shape))
    # cv_show(down)
    up2 = cv2.pyrUp(img)
    print("up2 shape:{}".format(up2.shape))
    # cv_show(up2)
    down2 = cv2.pyrDown(down)
    print("down2 shape:{}".format(down2.shape))
    # cv_show(down2)

    upDown = cv2.pyrDown(up)
    print("upDown shape:{}".format(upDown.shape))
    cv_show(upDown)
    downUp = cv2.pyrUp(down) # 还原回来很模糊， 下采样损失很多信息
    print("downUp shape:{}".format(downUp.shape))
    cv_show(downUp)

def laplasPyramidTest():
    """
    拉普拉斯金字塔：
        Li = Gi - PyrUp(PyrDown(Gi)) #
        G(i+1) = pyrDown(Gi)
        1.低通滤波  2.缩小尺寸 3. 放大尺寸 4. 图像相减

    :return:
    """
    img = cv2.imread("./images/AM.png")
    down = cv2.pyrDown(img)
    downUp = cv2.pyrUp(down)
    G = img - downUp
    cv_show(G)


if __name__=="__main__":
    # gaosiPyramidTest()
    laplasPyramidTest()

