# -*- coding:utf-8 -*-
import cv2
import sys
image_path = r"./nane.jpg"


def open_image():
    image = cv2.imread(image_path) # 返回的是numpy
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=="__main__":
    image = cv2.imread(image_path)
    print(type(image))
    print(image)
