# openCV_python
learn about openCV in python<br>
python 环境： python\==3.6.2， opecv\==3.4.1.15<br>
openCV安装：<br>
1. 安装opencv-python<br>
    opencv在3.4.2以后有些算法申请了专利，不能调用了；故我们安装 3.4.1<br>
    pip install \--user opencv-python==3.4.1.15<br> 速度慢就换源<br>
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


