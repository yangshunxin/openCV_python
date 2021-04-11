# openCV_python
learn about openCV in python<br>
环境安装
python 环境： python\==3.6.2， opecv\==3.4.1.15<br>
openCV安装：<br>
1. 安装opencv-python<br>
    opencv在3.4.2以后有些算法申请了专利，不能调用了；故我们安装 3.4.1<br>
    pip install \--user opencv-python==3.4.1.15  速度慢就换源<br>
    pip install \--user -i http://pypi.douban.com/simple \--trusted-host pypi.douban.com opencv-python\==3.4.1.15<br>
    测试安装成功<br>
    import cv2<br>
    cv2.\_\_version\_\_<br>
    输出:<br>
    3.4.1  就对了<br>
2. 安装额外的扩展 opencv-contrib-python<br>
    3版本后会有这个东东<br>
    安装版本必须与1中的一样，命令:<br>
    pip install --user opencv-contrib-python==3.4.1.15 # 速度慢就换源<br>
    pip install --user -i http://pypi.douban.com/simple --trusted-host pypi.douban.com opencv-contrib-python==3.4.1.15<br>
或者在直接下载openCV的包来安装：https://www.lfd.uci.edu/~gohlke/pythonlibs/<br>


内容说明<br>
cvBaseOperation.py: 关于图片的基本操作：图片和视频的打开，显示和保存，以及切片ROI，数值计算，通道分离，添加边界等操作，还有多图融合；<br>
    thresholdAndSmothOperation.py: 图像二值化，平滑处理（低通滤波）：均值滤波，方框滤波，高斯滤波和终止滤波；<br>
    morphologyOperation.py: 形态学操作，包括腐蚀，膨胀，开运算，闭运算，计算梯度，顶帽和黑帽<br>
    gradientCalcOperation.py: 梯度计算，包括sobel算子，scharr算子，拉普拉斯算子<br>
    cannyEdgeDetectionOperation.py: canny算法的具体步骤，包括梯度的求导，线性插值，NMS等操作<br>
    pyramidOperation.py: 图像金字塔，如何生成的，包括高斯金字塔和拉普拉斯金字塔<br>
    contourDetectionOperation.py：轮廓的检测和轮廓的截取，以及模板匹配（找到相似的区块）--类似卷积<br>
    histAndEqualizationOperation.py：直方图的计算，以及均衡化<br>
    FourierTransformOperation.py：傅里叶变换的原理，FFT，高通滤波和低通滤波<br>


项目说明<br>
    ocr_template_match：用模板匹配的方法，实现对信用卡数字的识别；先用形态学的方法实现数字块的分割，再用宽高比 和 宽高像素等方法，得到数字块，最后对每个块分割出每个数字块，跟模板的10个数字块进行匹配；<br>

