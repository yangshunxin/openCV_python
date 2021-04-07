# -*- coding:utf-8 -*-
import cv2, time
import numpy as np

def show_image(np_image):
    cv2.imshow("image", np_image)
    # cv2.waitKey(0)
    time.sleep(5)
    cv2.destroyAllWindows()

if __name__=="__main__":
    """
    virtual_image = np.zeros([480, 480], np.uint8)
    virtual_image = np.ones([480, 480], np.uint8)
    virtual_image = np.fromiter([255 for x in range(480*480*3)], np.uint8).reshape([480, 480, 3])
    show_image(virtual_image)
    print(virtual_image)
    """

