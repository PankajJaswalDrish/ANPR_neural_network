import pytesseract
import cv2
import numpy as np


def image_preprocessing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray,5)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    kernel = np.ones((5,5),np.uint8)
    dilate = cv2.dilate(blur, kernel, iterations = 1)
    text = pytesseract.image_to_string(thresh, config ='--psm 6')
    print(text)
    cv2.imshow("dilate", dilate)
    cv2.waitKey(0)
    return image

img = cv2.imread('testocr.png')
image_preprocessing(img)
    
'''kernel = np.ones((5,5),np.uint8)
erode = cv2.erode(dilate, kernel, iterations = 1)

kernel = np.ones((5,5),np.uint8)
new_image = cv2.morphologyEx(erode, cv2.MORPH_OPEN, kernel)
canny = cv2.Canny(new_image, 100, 200)'''
'''
text = pytesseract.image_to_string(thresh, config ='--psm 6')
print(text)

cv2.imshow("result", thresh)
cv2.waitKey(0)'''