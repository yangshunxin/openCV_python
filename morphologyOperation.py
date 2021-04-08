import cv2
import numpy as np
import matplotlib.pyplot as plt
from cvBaseOperation import cv_show

def erodeTest():
    img = cv2.imread("./images/dige.png")
    print(img.shape)
    # cv_show(img)
    kernel = np.ones((5, 5), np.uint8)
    # 总共有两个参数： kernel 和 循环次数；
    # 腐蚀会使白色变小
    erosion = cv2.erode(img, kernel, iterations=1) # 白字变细了, 胡须没有了
    cv_show(erosion)

def erodeTest2():
    img = cv2.imread("./images/pie.png")
    print(img.shape)
    # cv_show(img)
    kernel = np.ones((30, 30), np.uint8)
    erosion_1 = cv2.erode(img, kernel, iterations=1)
    erosion_2 = cv2.erode(img, kernel, iterations=2)
    erosion_3 = cv2.erode(img, kernel, iterations=3)
    res = np.hstack((erosion_1, erosion_2, erosion_3))
    cv_show(res)

def dilateTest():
    img = cv2.imread("./images/dige.png")
    print(img.shape)
    # cv_show(img)
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=1)
    dige_dilate = cv2.dilate(erosion, kernel, iterations=1)  # 白色被放大
    cv_show(dige_dilate)

def dilateTest2():
    img = cv2.imread("./images/pie.png")
    print(img.shape)
    # cv_show(img)
    kernel = np.ones((30, 30), np.uint8)
    dilate_1 = cv2.dilate(img, kernel, iterations=1)
    dilate_2 = cv2.dilate(img, kernel, iterations=2)
    dilate_3 = cv2.dilate(img, kernel, iterations=3)
    res = np.hstack((dilate_1, dilate_2, dilate_3))
    cv_show(res)

def openMorphTest():
    # 开运算： 先腐蚀，再膨胀
    # 没有胡须
    img = cv2.imread("./images/dige.png")
    kernel = np.ones((5, 5), np.uint8)

    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    cv_show(opening)

def closeMorphTest():
    # 闭运算： 先膨胀，再腐蚀
    #　膨胀后　胡须变大，再腐蚀，就会腐蚀失败
    img = cv2.imread("./images/dige.png")

    kernel = np.ones((5, 5), np.uint8)

    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    cv_show(closing)

def gradientTest():
    # 梯度 = 膨胀 - 腐蚀
    img = cv2.imread("./images/pie.png")
    kernel = np.ones((7, 7), np.uint8)

    # 膨胀
    dilate = cv2.dilate(img, kernel, iterations=1)
    # 腐蚀
    erosion = cv2.erode(img, kernel, iterations=1)

    res = np.hstack((dilate, erosion))

    gradient = cv2.subtract(dilate, erosion)
    cv_show(gradient)

def gradientTest2():
    img = cv2.imread("./images/pie.png")
    kernel = np.ones((7, 7), np.uint8)
    # 计算梯度
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

    cv_show(gradientTest())

def topHatTest():
    # 顶帽 = 原始输入 - 开运算结果
    # 只有细节信息
    img = cv2.imread("./images/dige.png")
    kernel = np.ones((7, 7), np.uint8)
    topHat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    cv_show(topHat)

def blackHatTest():
    # 黑帽 = 闭运算 - 原始输入
    # 原始的小轮廓
    img = cv2.imread("./images/dige.png")
    kernel = np.ones((7, 7), np.uint8)
    blackHat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    cv_show(blackHat)


if __name__=="__main__":
    # erodeTest()
    # erodeTest2()
    # dilateTest()
    # dilateTest2()
    # openMorphTest()
    # closeMorphTest()
    # gradientTest()
    # gradientTest2()
    # topHatTest()
    blackHatTest()