import cv2
import numpy as np
import matplotlib.pyplot as plt
from cvBaseOperation import cv_show



def sobelTest():
    """
    sobel算子， 这里是X方向的算子， 右-左
    Gx = [[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]]

    Y方向的算子， 下 - 上
    Gy = [[-1, -2, -1],
          [0, 0, 0],
          [1, 2, 1]]
    dst = cv2.Sobel(src, ddepth, dx, dy, ksize)
    ddepth: 图像的深度
    dx和dy分别表示水平和竖直方向; 可以单独计算也可以一起计算，建议分开计算再合并
    ksize是Sobel算子的大小

    """
    # img = cv2.imread("./images/pie.png", cv2.IMREAD_GRAYSCALE)
    img = cv2.imread("./images/lena.jpg", cv2.IMREAD_GRAYSCALE)
    # cv_show(img)
    # 先计算X方向
    # 白到黑是正数，黑到白是负数，所有的负数会被截断成0， 索要要取绝对值
    # cv2.CV_64F 能表示负数
    sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    # cv_show(sobelX)
    sobelX = cv2.convertScaleAbs(sobelX) # 但是上面会缺失
    # cv_show(sobelX)

    # 再计算Y方向
    sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobelY = cv2.convertScaleAbs(sobelY)
    # cv_show(sobelY)

    # 将X和Y方向合并
    sobelXY = cv2.addWeighted(sobelX, 0.5, sobelY, 0.5, 0)
    cv_show(sobelXY)

    # 直接一起计算
    # 有重影，实际用的时候不建议一起计算
    sobelXYTogether = cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize=3)
    sobelXYTogether = cv2.convertScaleAbs(sobelXYTogether)
    # cv_show(sobelXYTogether)

def scharrAndLaplacianTest():
    """
    图像梯度：
    Scharr算子： 比sobel算子更精细，
    Gx = [[-3, 0, 3],
          [-10, 0, 10],
          [-3, 0, 3]]
    Gy = [[-3, -10, -3],
          [0, 0, 0],
          [-3, -10, -3]]
    laplacian算子： 二阶导，对变化更明显， 但是对噪音敏感
    G = [[0, 1, 0],
         [1, -4, 1],
         [0, 1, 0]]
    :return:
    """
    # img = cv2.imread("./images/pie.png", cv2.IMREAD_GRAYSCALE)
    img = cv2.imread("./images/lena.jpg", cv2.IMREAD_GRAYSCALE)
    sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobelX = cv2.convertScaleAbs(sobelX)
    sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobelY = cv2.convertScaleAbs(sobelY)
    sobelXY = cv2.addWeighted(sobelX, 0.5, sobelY, 0.5, 0)

    scharrX = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    scharrY = cv2.Scharr(img, cv2.CV_64F, 0, 1)
    scharrX = cv2.convertScaleAbs(scharrX)
    scharrY = cv2.convertScaleAbs(scharrY)
    scharrXY = cv2.addWeighted(scharrX, 0.5, scharrY, 0.5, 0)

    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)

    res = np.hstack((sobelXY, scharrXY, laplacian))
    cv_show(res)


if __name__=="__main__":
    # sobelTest()
    scharrAndLaplacianTest()
