import cv2
import matplotlib.pyplot as plt
import numpy as np

def imageBaseTest():
    # 默认读取的是彩色图， 读取的格式
    # cv2.IMREAD_COLOR：彩色图像
    # cv2.IMREAD_GRAYSCALE：灰度图像

    img = cv2.imread("./images/cat.jpg")
    # img = cv2.imread("./images/cat.jpg", cv2.IMREAD_GRAYSCALE)
    # color shape: (414, 500, 3) 分别为(height, width, channels)
    # Grayscale shape: (414, 500) 分别为(height, width)
    print(img.shape)
    print(type(img)) # numpy.ndarray
    # Grayscale图像的大小 207000， color图像的大小：621000
    print(img.size)
    cv_show(img, "image")

    cv_save("my_cat.png", img)


def cv_show(img, name="show"):
    # 图像的显示，也可也i创建多个窗口
    cv2.imshow(name, img)
    cv2.moveWindow(name, 40, 50)
    # 等待时间，毫秒级， 0表示任意健终止
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def cv_save(name, img):
    # 将img 保存到 name
    cv2.imwrite(name, img)

def getROICV():
    img = cv2.imread("./images/cat.jpg")
    cat = img[0:100, 0:400]
    cv_show("cat", cat)

def getSplitChannels():
    img = cv2.imread("./images/cat.jpg")
    # 按通道分开
    b, g, r = cv2.split(img) # 也可以用 numpy的切片
    # 三个shape都是一样的，(414, 500)
    print(b.shape)
    print(g.shape)
    print(r.shape)
    # 组合在一起
    img2 = cv2.merge((b, g, r))
    print(img2.shape)
    # cv_show(img2)

    # 只保留G
    cur_image = img.copy()
    cur_image[:, :, 0] = 0
    cur_image[:, :, 2] = 0
    # print(cur_image.shape)
    # cv_show(cur_image, "G")

    # 只保留B
    cur_image = img.copy()
    cur_image[:, :, 1] = 0
    cur_image[:, :, 2] = 0
    print(cur_image.shape)
    cv_show(cur_image, "B")

def borderFill():
    """
    BORDER_REPLICATE: 复制法，也就是复制最边缘像素
    BORDER_REFLECT: 反射法，对感兴趣的图像中的像素在两边进行复制，例如：fedcba|abcdefgh|hgfedcb
    BORDER_REFLECT_101: 反射法，也就是以最边缘像素为轴，对称，gfedcb|abcdefgh|gfedcba
    BORDER_WRAP: 外包装法 cdefgh|abcdefgh|abcdefg
    BORDER_CONSTANT: 常量法，常数值填充
    :return:
    """
    img = cv2.imread("./images/cat.jpg")
    top_size, bottom_size, left_size, right_size = (50, 50, 50, 50)
    replicate = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REPLICATE)
    reflect = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REFLECT)
    reflect101 = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REFLECT_101)
    wrap = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_WRAP)
    constant = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_CONSTANT, value=0)

    plt.subplot(231), plt.imshow(img, "gray"), plt.title("ORIGINAL")
    plt.subplot(232), plt.imshow(replicate, "gray"), plt.title("REPLICATE")
    plt.subplot(233), plt.imshow(reflect, "gray"), plt.title("REFLECT")
    plt.subplot(234), plt.imshow(reflect101, "gray"), plt.title("REFLECT_101")
    plt.subplot(235), plt.imshow(wrap, "gray"), plt.title("WRAP")
    plt.subplot(236), plt.imshow(constant, "gray"), plt.title("CONSTANT")

    plt.show()

def valueCalculate():
    img_dog = cv2.imread("./images/dog.jpg")
    img_cat = cv2.imread("./images/cat.jpg")

    # 每个像素点的每个通道都加10
    # print(img_cat[:2, :1, :1])  # 只显示前2行
    # print("="*5)
    img_cat2 = img_cat + 10
    # print(img_cat2[:2, :1, :1]) # 只显示前2行

    # shape 不同，不能直接相加
    # 每个像素对应的通道的值相加，当超过255时，对256取余数
    imgAdd = img_cat + img_cat2
    # print(imgAdd[:2, :1, :1])
    cv_add = cv2.add(img_cat, img_cat2)
    # print(cv_add[:2, :1, :1])
    # 猫和狗的图片融合
    # 先对dog进行resize
    print(img_dog.shape)
    img_dog = cv2.resize(img_dog, (500, 414)) # resize的第一种方式
    # cv2.resize(img_dog, (0, 0), fx=1, fy=3) # 表示沿着y轴方向拉升3倍
    print(img_dog.shape)
    print(img_cat.shape)
    res = cv2.addWeighted(img_cat, 0.4, img_dog, 0.6, 0)
    plt.imshow(res)
    # cv_show(res)

def videoBaseTest():
    # cv2.VideoCapture可以捕获摄像头，用数字来控制不同的设备，例如0，1
    #　如果是视频文件，直接指定好路径即可
    # vc = cv2.VideoCapture("./videos/test.mp4")
    vc = cv2.VideoCapture(0)
    # 检查是否打开正确
    if vc.isOpened():
        open, frame = vc.read() # 读取一帧数据， open：读取是否成功，frame:表示成功后的数据ndArray
    else:
        open = False

    while open:
        ret, frame = vc.read() # 读取一帧
        if frame is None:
            # 读取完了，退出循环
            break
        if ret == True:
            # 读取成功了一帧数据
            # 转成灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # gray = frame # 显示彩色
            # 显示在result窗口上
            cv2.imshow("result", gray)
            # 等待超过10毫秒，或者收到 esc键 退出
            if cv2.waitKey(10) & 0xFF == 27:
                break
    vc.release()
    cv2.destroyAllWindows()



if __name__=="__main__":
    # imageBaseTest()
    # videoBaseTest()
    # getROICV()
    # getSplitChannels()
    # borderFill()
    valueCalculate()
    pass

