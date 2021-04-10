import cv2
import numpy as np
import matplotlib.pyplot as plt
from cvBaseOperation import cv_show


def contourTest():
    """
    边缘与轮廓的区别：
        边缘可以是线段零零散散的，轮廓必须连接在一起整体；
     cv2.findContours(img, mode, method)
     mode: 轮廓检索模式
        RETR_EXTERNAL: 只检索最外面的轮廓；
        RETR_LIST: 检索所有的轮廓，并将其保存到一个链表中
        RETR_CCOMP: 检索所有的轮廓，并将他们组织为两层，顶层是各部分的外部边界，第二层是空洞的边界；
        RETR_TREE: 检索所有的轮廓，并重构嵌套轮廓的整个层次；--一般填这个
    method: 轮廓逼近方法
        CHAIN_APPROX_NONE：以Freeman链码的方式输出轮廓，所有其他的方法输出多边形（顶点的序列）
        CHAIN_APPROX_SIMPLE：压缩水平的、垂直的和斜的部分，也就是，函数只保留他们的终点部分。
    """
    # img = cv2.imread("./images/contours2.png")
    img = cv2.imread("./images/contours.png")
    # img = cv2.imread("./images/car.png")
    # 为了更高的准确率，使用二值图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # cv_show(thresh, "thresh")

    binary, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # binary跟输入的thresh一模一样，
    # contours是一个列表，表示检测的结果
    # hierarchy 层级

    #　绘制轮廓
    # 传入绘制图像，轮廓，轮廓索引，颜色模式，线条厚度
    # 注意需要copy, 要不原图会变
    draw_img = img.copy()
    # draw_img = thresh.copy() # 灰度图看不清
    res = cv2.drawContours(draw_img, contours, -1, (0, 0, 255), 2)
    # cv_show(res)

    #　计算轮廓的面积和周长
    index = 0
    cnt = contours[index]
    area = cv2.contourArea(cnt)
    # True: 表示闭合的
    arcLen = cv2.arcLength(cnt, True)
    print("index:{} area:{} 周长：{}".format(index, area, arcLen))

    # 轮廓近似
    epsilon = 0.2 * cv2.arcLength(cnt, True) # 阈值用 周长的0.1，可以变
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    draw_img2 = img.copy()
    res2 = cv2.drawContours(draw_img2, [approx], -1, (0, 0, 255), 2)
    # cv_show(res2)

    # 边界矩形
    x, y, w, h = cv2.boundingRect(cnt)
    imgRect = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    rectArea = w*h
    extent = float(area)/float(rectArea)
    print("轮廓面积与边界矩形比：{}".format(extent))

    # cv_show(imgRect)

    # 边界圆
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    imgCircle = cv2.circle(img, center, radius,(0, 255, 0), 2)
    cv_show(imgCircle)

def contourTemplateMatchTest():
    """
    模板匹配：
        模板匹配和卷积原理很像，模板在原图像上从原点开始滑动，计算模板与（图像被模板覆盖的地方）的差别程度，
    这个差别程度的计算方法在openCV里面有六种，计算后将每次计算的结果放入一个矩阵里，作为结果输出。
        假如原图形是A*B大小，则模板是a*b大小，则输出的结果饿矩阵就是（A-a+1）*（B-b+1）

    TM_SQDIFF: 计算平方不同，计算出来的值越小，越相关
    TM_CCORR: 计算相关性，计算出来的值越大，越相关
    TM_CCOEFF: 计算相关系数，计算出来的值越大，越相关
    TM_SQDIFF_NORMED：计算归一化平方不同，计算出来的值越接近0，越相关
    TM_CCORR_NORMED：计算归一化相关性，计算出来的值越接近1，越相关
    TM_CCOEFF_NORMED：计算归一化相关系数，计算出来的值越接近1，越相关
    一般使用归一化的参数
    公式： https://docs.opencv.org/3.3.1/df/dfb/group__imgproc__object.html#ga3a7850640f1fe

    :return:
    """
    img = cv2.imread("./images/lena.jpg", 0)
    template = cv2.imread("./images/lena_face.jpg", 0)
    print(img.shape)
    print(template.shape)
    h, w = template.shape[:2]
    res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF)
    print("res shape:{}".format(res.shape)) # (161, 194)

    # 获取结果中的最大值和最小值
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print(min_val, min_loc, max_val, max_loc)

    # 查看6中的结果
    methods = ["cv2.TM_SQDIFF", "cv2.TM_CCORR", "cv2.TM_CCOEFF",
               "cv2.TM_SQDIFF_NORMED", "cv2.TM_CCORR_NORMED", "cv2.TM_CCOEFF_NORMED"]
    for meth in methods:
        img2 = img.copy()

        # 匹配方法的真值
        method = eval(meth)
        print(method)
        res = cv2.matchTemplate(img, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # 如果是平方差匹配TM_SQDIFF或归一化平方差匹配TM_SQDIFF_NORMED， 取最小值
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        # 画矩形
        cv2.rectangle(img2, top_left, bottom_right, 255, 2)

        plt.subplot(121), plt.imshow(res, cmap="gray")
        plt.xticks([]), plt.yticks([]) # 隐藏坐标轴
        plt.subplot(122), plt.imshow(img2, cmap="gray")
        plt.xticks([]), plt.yticks([]) # 隐藏坐标轴
        plt.suptitle(meth)
        plt.show()

def contourTemplateMultiMatchTest():
    # 匹配多个对象
    img = cv2.imread("./images/lena.jpg", 0)
    template = cv2.imread("./images/lena_face.jpg", 0)
    print(img.shape)
    print(template.shape)
    h, w = template.shape[:2]

    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    # 取匹配程度大于0.8的坐标
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]): #　*号表示可选参数
        bottom_right = (pt[0] + w, pt[1] + h)
        cv2.rectangle(img, pt, bottom_right, (0, 0, 255), 2)
    cv_show(img)

if __name__=="__main__":
    # contourTest()
    # contourTemplateMatchTest()
    contourTemplateMultiMatchTest()