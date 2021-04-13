import cv2
import numpy as np
import matplotlib.pyplot as plt
from cvBaseOperation import cv_show


def harrisTest():
    """
    cv2.cornerHarris()
    img: 数据类型为float32的输入图像
    blockSize: 角点检测中指定区域的大小
    ksize: Sobel求导中使用的窗口大小, 一般为3
    k: 取值参数为[0.04, 0.06]， 一般为0.04
    """
    # img = cv2.imread("./images/chessboard.jpg")
    img = cv2.imread("./images/test_1.jpg")
    print("img shape:{}".format(img.shape))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # grapy = np.float32(gray) # uint8 -> float32
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    print("dst shape:{}".format(dst.shape)) # 输出的是 每个点出的 值，故维度为(height, widith)
    img[dst > 0.01*dst.max()] = [0, 0, 255] # 将最大值的k倍， 将点标记为红色
    cv_show(img)



if __name__=="__main__":
    harrisTest()

