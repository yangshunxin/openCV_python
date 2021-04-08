import cv2
import numpy as np
import matplotlib.pyplot as plt
from cvBaseOperation import cv_show

"""
Canny边缘检测：
    1. 使用高斯滤波器，以平滑图像，滤除噪声
    2. 计算图像中每个像素点的梯度强度和方向
    3. 应用非极大值抑制(Non-Maximum Suppression), 以消除边缘检测带来的杂散响应
    4. 应用双阈值（Double-Threshold）检测来确定真实的和潜在的边缘
    5. 通过抑制孤立的弱边缘最终完成边缘检测
    
    1. 高斯滤波器
        H = [[0.0924, 0.1192, 0.0924], 
             [0.1192, 0.1538, 0.1192], # 这里做了归一化处理
             [0.0924, 0.1192, 0.0924]]
        e = H*A
    2. 计算梯度和方向
        通过sobel算子，分别计算出X方向和Y方向的梯度，
        然后 G = (Gx^2 + Gy^2)^0.5 计算每个点的总梯度
        方向 θ = arctan(Gy/Gx)
        其中Gx和Gy的计算见梯度计算篇(gradientCalcOperation)
        sobel算子:
            Sx = [[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]]
        
            Sy = [[-1, -2, -1],
                  [0, 0, 0],
                  [1, 2, 1]]
    3.非极大值抑制
        原理：
        指的是找像素点局部最大值，将非极大值点所对应的灰度值置为0，这样可以剔除掉一大部分非边缘的点。
        确认像素点C的灰度值在其8领域内是否为最大；如何确定昵？找到C点的梯度方向的直线，
        那么极大值点肯定在这条直线上， 这条线与外八点框的交点tmp1和tmp2，极值点也可能是tmp1和tmp2;
        因此判断C点灰度与这两个点灰度的大小即可判断C点是否为其领域内的局部最大灰度点。如果经过判断，
        C点灰度值小于这两个点中的任一个，那就说明C点不是局部最大值，那么排除C点为边缘，这就是非极大值抑制的原理。
        
        问题1：非极大值抑制回答了这样一个问题：当前的梯度值在梯度方向上是一个局部最大值吗！！！所以要将当前位置的梯度值
        与梯度方向上两侧的梯度值进行比较。
        
        问题2：梯度方向垂直于边缘方向。实际上我们只能得到C点领域的8个点的值，而tmp1和tmp2并不在其中，
        要得到这两个值就需要对该两个点两端的抑制灰度进行线性插值，得到tmp1和tmp2的灰度值，还要用到其梯度方向，
        这是Canny算法中要求解梯度方向矩阵Thita的原因。
        
        结果：完成非极大值抑制后，会得到一个二值图像，非边缘的点灰度值均为0，可能为边缘的局部灰度极大值点可设置为维度为128.
        检测结果还包含了很多由噪声及其他原因造成的假边缘，还要做进一步处理。
        非极大值抑制算法（Non-maximum suppression, NMS）的本质是搜索局部极大值，抑制非极大值元素，跟最近的两个点做比较。
        参看链接：https://www.zhihu.com/question/37172820
        
        还有一种算法就是判断C点的梯度方向离周边8个点哪个近就取那个点来跟C进行比较，就不用插值了。
    
    4.双阈值检测：
        给到两个阈值：maxVal和minVal；
            1. 如果梯度值>maxVal,则处理为边界；
            2. 如果minVal<梯度值<maxVal：连有边界则保留，否则舍弃；（连有边界的意思是旁边是边界点，即1中的点）
            3. 梯度值<minVal: 则舍弃
"""
def cannyTest():
    img = cv2.imread("./images/lena.jpg", cv2.IMREAD_GRAYSCALE)
    v1 = cv2.Canny(img, 80, 150)
    v2 = cv2.Canny(img, 50, 100) # 会获得更多的细节边界

    res = np.hstack((v1, v2))
    cv_show(res)

def cannyTest2():
    img = cv2.imread("./images/car.png", cv2.IMREAD_GRAYSCALE)
    v1 = cv2.Canny(img, 120, 250)
    v2 = cv2.Canny(img, 50, 100) # 会获得更多的小细节边界

    res = np.hstack((v1, v2))
    cv_show(res)

if __name__=="__main__":
    # cannyTest()
    cannyTest2()