import cv2
import numpy as np
import matplotlib.pyplot as plt
from cvBaseOperation import cv_show


def dftTest():
    """
    什么是傅里叶变换？
        傅里叶同学告诉我们，任何周期函数，都可以看作是不同振幅，不同相位正弦波的叠加。
        而贯穿时域与频域的方法之一，就是传中说的傅里叶分析。
        傅里叶分析可分为傅里叶级数（Fourier Serie）和傅里叶变换(Fourier Transformation)；
        什么是相位谱：时间差并不是相位差。如果将全部周期看作2Pi或者360度的话，相位差则是时间差在一个周期中所占的比例。我们将时间差除周期再乘2Pi，就得到了相位差，最大为2Pi
        傅里叶级数的本质是将一个周期的信号分解成无限多分开的（离散的）正弦波；
        傅里叶级数，在时域是一个周期且连续的函数，而在频域是一个非周期离散的函数;
        傅里叶变换，则是将一个时域非周期的连续信号，转换为一个在频域非周期的连续信号******很重要
        傅里叶变换实际上是对一个周期无限大的函数进行傅里叶变换
        欧拉公式： e^(ix) = cos(x) + i*sin(x)
        e^(it)可以理解为一条逆时针旋转的螺旋线
        对傅里叶变换的理解参考：https://zhuanlan.zhihu.com/p/19763358 （傅里叶分析之掐死教程）
        对傅里叶级数的证明参考：https://zhuanlan.zhihu.com/p/41455378

    应用：
        任何连续的周期信号可以分解成许多正弦波与余弦波；
    为什么分解成正余弦波？
        因为正余弦波输入一个系统，输出仍然是正余弦波，只有幅度与相位发生变化，频率与波形保持不变，更容易分析信号特征。

    傅里叶变换分为四种：
        1. 非周期连续信号———->傅里叶变换（FourierTransform） 例如：高斯分布曲线
        2. 周期连续信号———–>傅里叶级数 （Fourier Series） 例如：正弦波，方波
        3. 非周期离散信号———>离散时间域的傅里叶变换（Discrete TimeFourierTransform）DTFT
        4. 周期离散信号———->离散傅里叶级数（DiscreteFourierSeries）但一般都叫（Discrete Fourier Transform）DFT

    实际用于DSP的只有DFT（离散傅里叶变换）
    傅里叶变换的作用：
        高频： 变化剧烈的灰度分量，例如边界
        低频： 变化缓慢的灰度分量，例如一片大海
        滤波：
            低通滤波器：只保留低频，会使图像模糊
            高频滤波器：只保留高频，会使图像细节增强
        opencv中主要就是cv2.dft()和cv2.idft(),输入图像需要先转换成np.float32格式
        得到结果中频率为0的部分会在左上角(四个角落)，需要转到中心位置，可以通过shift变换来实现
        cv2.dft()返回的结果是双通道的（实部和虚部），通常还需要转换成图像格式才能展示(0~255)

    """
    img = cv2.imread("./images/lena.jpg", 0)
    img_float32 = np.float32(img)

    dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT) # 这个就是dft的频谱图
    origin_magnitude_spectrum = 20 * np.log(cv2.magnitude(dft[:, :, 0], dft[:, :, 1]))
    print(origin_magnitude_spectrum.shape)
    # print(origin_magnitude_spectrum[:, 0])
    # print(origin_magnitude_spectrum[:, 262])
    dft_shift = np.fft.fftshift(dft) # 中心点在左上角，变到中心点
    # 得到灰度图能表示的形式
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    plt.subplot(131), plt.imshow(img, cmap="gray")
    plt.title("input image"), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(origin_magnitude_spectrum, cmap="gray")
    plt.title("Origin "), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(magnitude_spectrum, cmap="gray")
    plt.title("Magnitude Spectrum"), plt.xticks([]), plt.yticks([])
    plt.show()

def filterRejectorTest():
    img = cv2.imread("./images/lena.jpg", 0)
    img_float32 = np.float32(img)

    dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT) # 这个就是dft的频谱图
    dft_shift = np.fft.fftshift(dft) # 中心点在左上角，变到中心点
    # 得到灰度图能表示的形式
    # magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    rows, cols = img.shape
    crow, ccol = int(rows/2), int(cols/2) # 中心位置

    # 低通滤波器 --图像变模糊了
    mask = np.zeros((rows, cols, 2), np.int8)
    mask[crow-30:crow+30, ccol-30:ccol+30] = 1 # 保留中间的低频部分

    # 高通滤波器
    maskUp = np.ones((rows, cols, 2), np.int8)
    maskUp[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0

    # IDFT
    # fshitf = dft_shift*mask # 滤波过程
    fshitf = dft_shift*maskUp # 滤波过程
    f_ishift = np.fft.ifftshift(fshitf) # 将中心转到四周
    img_back = cv2.idft(f_ishift) # 逆变换
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    plt.subplot(121), plt.imshow(img, cmap="gray")
    plt.title("input image"), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img_back, cmap="gray")
    plt.title("Result"), plt.xticks([]), plt.yticks([])
    plt.show()


if __name__=="__main__":
    # dftTest()
    filterRejectorTest()