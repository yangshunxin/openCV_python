from imutils import contours
import numpy as np
import cv2
from ocr_template_match import myutils

# 设置参数
# import argparse
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, default="", help="path to input image")
# ap.add_argument("-t", "--template", required=True, default="", help="path to input template OCR-A image")
#
# args = vars(ap.parse_args())

# 指定信用卡类型
FIRST_NUMBER = {
    "3": "American Express",
    "4": "Visa",
    "5": "MasterCard",
    "6": "Discover Card"
}

bShow = True
bShow = False

# 绘制图片
def cv_show(img, name="img"):
    if bShow:
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def reprocessTemplate(templatePath):
    # 读取模板图像
    img = cv2.imread(templatePath)
    cv_show(img, "template")
    # 灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv_show(gray)
    # 二值化
    binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)[1]
    cv_show(binary, "binary")

    #　计算轮廓，
    #　findContokurs()函数接收的参数为二值图，即黑白的（不是灰度图），cv2.RETR_EXTERNAL只检测外轮廓，cv2.CHAIN_APPROX_SIMPLE只保留终点坐标
    # 返回的list中每个元素都是图像中的一个轮廓
    ref_, refCnts, hierarchy = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, refCnts, -1, (0, 0, 255), 3)
    cv_show(img)
    print(np.array(refCnts).shape)
    refCnts = myutils.sort_contours(refCnts, method="left-to-right")[0] # 排序， 从左到右， 从上到下
    digits = {}

    # 遍历每一个轮廓
    for (i, c) in enumerate(refCnts):
        # 计算外接矩形并且resize成合适大小
        (x, y, w, h) = cv2.boundingRect(c)
        roi = binary[y: y+h, x: x + w]
        roi = cv2.resize(roi, (57, 88)) # 放缩到同样的比例大小

        # 每一个数字对应每一个模板
        digits[i] = roi

    return img, digits

def reprocessSample(samplePath, templateDigitsDict):
    # 读取sample 并进行预处理

    # 初始化卷积核
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    imageOrigin = cv2.imread(sampleImagePath)
    # resize
    image = myutils.resize(imageOrigin, width=300)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv_show(gray)

    # 顶帽，突出更明亮的区域
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
    cv_show(tophat, "topHat")

    # 计算梯度，相当于轮廓提取
    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1) # ksize=-1 相当于3*3
    gradX = np.absolute(gradX)
    minVal, maxVal = np.min(gradX), np.max(gradX)
    gradX = (255 * ((gradX - minVal)/(maxVal - minVal))) # 把值 放缩到 0~255
    gradX = gradX.astype("uint8")

    print(np.array(gradX).shape)
    cv_show(gradX, "gradX")

    # 通过闭操作（先膨胀，再腐蚀）将数字连在一起---闭操作扩大白色区域
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    cv_show(gradX, "after close gradX")
    # 二值化，THRESH_OTSU会自动寻找合适的阈值，适合双峰，需把阈值参数设置为0
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv_show(thresh, "binary")
    # 再对二值化后 来一次闭操作，
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    cv_show(thresh, "two close gradx")

    thresh, threshCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = threshCnts
    cur_img = image.copy()
    cv2.drawContours(cur_img, cnts, -1, (0, 0, 255), 3)
    cv_show(cur_img, "contour detect")

    locs = []
    # 遍历所有的轮廓
    for (i, c) in  enumerate(cnts):
        # 计算矩形
        (x, y, w, h) = cv2.boundingRect(c) # 外接矩形
        ar = w / float(h) # 宽高比

        # 选择合适的区域，根据实际任务来，这里的基本都是四个数字一组
        if ar > 2.5 and ar < 4.0:
            if (w > 40 and w < 55) and (h > 10 and h < 20):
                # 符合条件留下来
                locs.append((x, y, w, h))

    # 将符合的轮廓从左到右排序--应该有4个块
    locs = sorted(locs, key=lambda x:x[0]) # x坐标来排序
    output = []

    # 遍历每一个轮廓中的数字
    for (i, (gX, gY, gW, gH)) in enumerate(locs):
        #　init the list of group digits
        groupOutput = []

        # 根据坐标提取每一个组
        group = gray[gY -5:gY + gH +5, gX -5: gX + gW + 5]
        cv_show(group)
        # 获取每一个数字，每组4个数字
        # 预处理
        group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # 计算每一组的轮廓
        group_, digitCnts, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]

        # 计算每一组中的每一个数值
        for c in digitCnts:
            # 找到当前数值的轮廓，resize成合适的大小
            (x, y, w, h) = cv2.boundingRect(c)
            roi = group[y:y + h, x: x + w]
            roi = cv2.resize(roi, (57, 88))
            cv_show(roi)

            # 计算匹配得分
            scores = []

            # 在模板中计算每一个得分
            for (digit, digitROI) in templateDigitsDict.items():
                # 匹配模板
                result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
                _, score, _, _ = cv2.minMaxLoc(result)
                scores.append(score)

            # 得到最合适的数字
            groupOutput.append(str(np.argmax(scores)))
        # 画出来
        cv2.rectangle(image, (gX -5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1),
        cv2.putText(image, "".join(groupOutput), (gX, gY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        # 得到结果
        output.extend(groupOutput)
    # 打印结果
    print("Credit Card Type: {}".format(FIRST_NUMBER[output[0]]))
    print("Credit Card #: {}".format("".join(output)))
    cv2.imshow("Image", image)
    cv2.waitKey(0)

def business(templatePath, samplePath):
    # templateDigitsDict, key: num,  value: template data,  data size (57, 88), len: 10 包括10个数
    templateImag, templateDigitsDict = reprocessTemplate(templatePath)

    reprocessSample(sampleImagePath, templateDigitsDict)




if __name__=="__main__":
    templatePath = r"../ocr_template_match/images/ocr_a_reference.png"
    # sampleImagePath = r"../ocr_template_match/images/credit_card_01.png"
    # sampleImagePath = r"../ocr_template_match/images/credit_card_02.png"
    # sampleImagePath = r"../ocr_template_match/images/credit_card_03.png"
    # sampleImagePath = r"../ocr_template_match/images/credit_card_04.png"
    sampleImagePath = r"../ocr_template_match/images/credit_card_05.png"

    business(templatePath, sampleImagePath)

