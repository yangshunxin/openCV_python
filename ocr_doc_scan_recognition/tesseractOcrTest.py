from PIL import Image
import pytesseract
import cv2
import os

# https://digi.bib.uni-mannheim.de/tesseract/
# 当前目录下带了一个4.0的版本
# 配置环境变量如E:\Program Files (x86)\Tesseract-OCR
# 配置windows环境遍历增加key-value: key:TESSDATA_PREFIX   value:C:\Program Files (x86)\Tesseract-OCR\tessdata
# tesseract -v进行测试
# tesseract XXX.png 得到结果
# pip install pytesseract
# anaconda lib site-packges pytesseract pytesseract.py
# tesseract_cmd 修改为绝对路径即可

# 免得配置环境 --很麻烦
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'

preprocess = 'blur'  # thresh

image = cv2.imread('scan.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

if preprocess == "thresh":
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

if preprocess == "blur":
    gray = cv2.medianBlur(gray, 3)

filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)

# 免得配置环境 --不然会报错，很麻烦
tessdata_dir_config = '--tessdata-dir "C:\\Program Files (x86)\\Tesseract-OCR\\tessdata"'

text = pytesseract.image_to_string(Image.open(filename), config=tessdata_dir_config)
print(text)
os.remove(filename)

cv2.imshow("Image", image)
cv2.imshow("Output", gray)
cv2.waitKey(0)