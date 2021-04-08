import cv2
import numpy as np
import matplotlib.pyplot as plt
from cvBaseOperation import cv_show

def thresholdTest():
    """
    retm dst = cv2.threshold(src, thresh, maxval, type)
        src: 输入图片，只能输入单通道图像，通常来说为灰度图
        dst: 输出图
        thresh: 阈值
        maxval: 当像素值超过了阈值（或者小于，根据type来定）， 所赋予的值
        type: 二值化操作的类型，包含以下5种类型：cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV,
        cv2.THRESH_TRUNC, cv2.THRESH_TOZERO, cv2.THRESH_TOZERO_INV;
        参数功能说明：
        cv2.THRESH_BINARY: 超过阈值部分取maxval(最大值)，否则取0
        cv2.THRESH_BINARY_INV: THRESH_BINARY的反转
        cv2.THRESH_TRUNC: 大于阈值的部分设为阈值，否则不变
        cv2.THRESH_TOZERO: 大于阈值部分不变，否则设为0
        cv2.THRESH_TOZERO_INV: THRESH_TOZERO的反转

    :return:
    """

    imgGray = cv2.imread("./images/dog.jpg", cv2.IMREAD_GRAYSCALE)

    ret, thresh1 = cv2.threshold(imgGray, 127, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(imgGray, 127, 255, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(imgGray, 127, 255, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(imgGray, 127, 255, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(imgGray, 127, 255, cv2.THRESH_TOZERO_INV)

    titles = ["orginal", "BINARY", "BINARY_INV", "TRUNC", "TOZERO", "TOZERO_INV"]
    images = [imgGray, thresh1, thresh2, thresh3, thresh4, thresh5]
    for i in range(6):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], "gray")
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

def smoothOperationTest():
    # 带有椒盐噪声的图像
    img = cv2.imread("./images/lenaNoise.png")
    print(img.shape)
    # cv_show(img)

    # 均值滤波
    # 简单的平均卷积操作
    blur = cv2.blur(img, (3, 3))

    # 方框滤波
    #　可以选择是否归一化，归一化就跟均值滤波一摸一样，不归一化　相加后最大值为255
    box = cv2.boxFilter(img, -1, (3, 3), normalize=True)
    # box = cv2.boxFilter(img, -1, (3, 3), normalize=False)
    # cv_show(box)

    # 高斯滤波
    # 高斯滤波的卷积核里的数值时满足高斯分布，相当于更重视中间的
    aussian = cv2.GaussianBlur(img, (5, 5), 1)
    # cv_show(aussian)

    # 中值滤波
    # 对卷积核中的数值进行排序，取中间的值取代
    median = cv2.medianBlur(img, 5)
    # cv_show(median)

    # 展示所有
    res = np.hstack((blur, aussian, median))
    cv_show(res)


if __name__=="__main__":
    # thresholdTest()
    smoothOperationTest()