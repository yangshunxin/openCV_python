# openCV_python
learn about openCV in python
python 环境： python 3.6.2
openCV安装：
1. 安装opencv-python
    opencv在3.4.2以后有些算法申请了专利，不能调用了；故我们安装 3.4.1
    pip install --user opencv-python==3.4.1.15 # 速度慢就换源
    pip install --user -i http://pypi.douban.com/simple --trusted-host pypi.douban.com opencv-python==3.4.1.15
    # 测试安装成功
    import cv2
    cv2.__version__
    输出：
    '3.4.1' # 就对了
2. 安装额外的扩展 opencv-contrib-python
    3版本后会有这个东东
    安装版本必须与1中的一样，命令：
    pip install --user opencv-contrib-python==3.4.1.15 # 速度慢就换源
    pip install --user -i http://pypi.douban.com/simple --trusted-host pypi.douban.com opencv-contrib-python==3.4.1.15


