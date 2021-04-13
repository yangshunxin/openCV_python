import cv2
import numpy as np
import matplotlib.pyplot as plt
from cvBaseOperation import cv_show


"""
    每个点会生成一个128维的特征向量；
    从opencv后的3.4.3开始，这个接口不能调用了，有专利
"""

def siftTest():
    img = cv2.imread("./images/test_1.jpg")
    # img = cv2.imread("./images/lena.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 得到特征点
    # 构造sift实例
    sift = cv2.xfeatures2d.SIFT_create()
    # 获取关键点
    kp = sift.detect(gray, None)

    # 画出关键点
    img = cv2.drawKeypoints(gray, kp, img)
    cv_show(img)

    # 计算特征
    kp, des = sift.compute(gray, kp)
    print(np.array(kp).shape) # 关键点的坐标 (6827,)
    print(des.shape) # (6827,128) 关键点对应的向量，向量维度是 128，

def siftFeatureMatchTest():
    smallImg = cv2.imread("./images/box.png", 0)
    img = cv2.imread("./images/box_in_scene.png", 0)

    cv_show(smallImg)
    cv_show(img)

    # 构建sift实例
    sift = cv2.xfeatures2d.SIFT_create()

    # 计算关键点和点对应向量
    kp1, des1 = sift.detectAndCompute(smallImg, None)
    kp2, des2 = sift.detectAndCompute(img, None)

    # 暴力匹配
    # crossCheck表示两个特征要相互匹配，
    # 例如A中的第i点的特征与B中的第j点的特征最近，并且B中的第j个特征点到A中第i个特征点也是最近滴
    bf = cv2.BFMatcher(crossCheck=True)

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x:x.distance)
    img3 = cv2.drawMatches(smallImg, kp1, img, kp2, matches[:10], None, flags=2)
    cv_show(img3)

    # K对最佳匹配
    bf2 = cv2.BFMatcher()
    matchesK = bf2.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matchesK:
        if m.distance < 0.75*n.distance:
            good.append([m])

    img3 = cv2.drawMatchesKnn(smallImg, kp1, img, kp2, good[:10], None, flags=2)

    cv_show(img3)


if __name__=="__main__":
    # siftTest()
    siftFeatureMatchTest()