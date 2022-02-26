import numpy as np
import cv2
import pytesseract
import matplotlib.pyplot as plt  
from PIL import Image
img = cv2.imread('8.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('img', img) # anh goc
thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
contours,h=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
largest_rectangle = [0,0]
for cnt in contours:
    approx = cv2.approxPolyDP(cnt,0.04*cv2.arcLength(cnt,True),True)
    if len(approx)==4: 
        area = cv2.contourArea(cnt)
        if area > largest_rectangle[0]:
            largest_rectangle = [cv2.contourArea(cnt), cnt, approx]
x,y,w,h = cv2.boundingRect(largest_rectangle[1])
roi=img[y:y+h,x:x+w]
cv2.drawContours(img,[largest_rectangle[1]],0,(255,255,255),4)
cv2.imshow('img1', thresh) # hinh anh phan nguong thich nghi
cv2.imshow('img2', roi) # hinh ảnh cat


#---------------
pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract.exe'
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3,3), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
cv2.imshow('CROP', thresh)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
invert = 255 - opening # dao nguoc mau trang thanh den
data = pytesseract.image_to_string(invert, lang='eng', config='--psm 6')
print("THONG TIN NHAN DIEN:")
print(data)
cv2.waitKey(0)
cv2.destroyAllWindows()

