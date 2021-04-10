import cv2
import numpy as np
import matplotlib.pyplot as plt
from cvBaseOperation import cv_show

def histTest():
    """
    cv2.calcHist(images, channels, mask, histSize, ranges)
    images: 原图像格式为uint8或float32。当传入参数时，应使用[], 例如[img]
    channels： 同样用中括号括起来，如果传入的是灰度图则传入[0], 如果是彩色图像则 可以传入[0]或[1]或[2]，他们分别对应BGR
    mask: 掩模图像。整幅图则设置为None，如果是整幅图的一部分，则填入掩模，见后面的例子
    histSize: BIN的数目，也用中括号括起来
    ranges: 像素值范围常为[0,256]
    """
    img = cv2.imread("./images/cat.jpg", 0) # 0表示灰度图
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    print(hist.shape)
    plt.hist(img.ravel(), 256)
    plt.show()

def histTest2():
    # 彩色图
    img = cv2.imread("./images/cat.jpg")
    color = ("b", "g", "r")
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()

def histTest3():
    # mask
    # 创建mask
    img = cv2.imread("./images/cat.jpg", 0)
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[100:300, 100:400] = 255
    # cv_show(mask)

    masked_img = cv2.bitwise_and(img, img, mask=mask) # 与操作
    # cv_show(masked_img)
    hist_full = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_mask = cv2.calcHist([img], [0], mask, [256], [0, 256])

    plt.subplot(221), plt.imshow(img, "gray")
    plt.subplot(222), plt.imshow(mask, "gray")
    plt.subplot(223), plt.imshow(masked_img, "gray")
    plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
    plt.xlim([0, 256])
    plt.show()

def equalTest():
    # img = cv2.imread("./images/cat.jpg", 0)
    # img = cv2.imread("./images/lena.jpg", 0)
    img = cv2.imread("./images/clahe.jpg", 0)
    # plt.hist(img.ravel(), 256)
    # plt.show()
    equ = cv2.equalizeHist(img)
    # plt.hist(equ.ravel(), 256)
    # plt.show()
    res = np.hstack((img, equ))
    # cv_show(res)
    # 自适应直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    res_clahe = clahe.apply(img)
    res2 = np.hstack((img, equ, res_clahe))
    cv_show(res2)

if __name__=="__main__":
    # histTest()
    # histTest2()
    # histTest3()
    equalTest()
