import cv2
import numpy as np

image = cv2.imread("4.jpg")

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

light_blue = np.array([110,50,50])
dark_blue = np.array([130,255,255])

mask = cv2.inRange(hsv, light_blue, dark_blue)
output = cv2.bitwise_and(image, image, mask= mask)

src1_mask=cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)#change mask to a 3 channel image 
mask_out=cv2.subtract(src1_mask, image)
mask_out=cv2.subtract(src1_mask, mask_out)

cv2.imshow("Color Detected", mask_out)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''import numpy as np
import cv2

# Color threshold
image = cv2.imread('4.jpg')
original = image.copy()
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([110,50,50])
upper = np.array([130,255,255])
mask = cv2.inRange(hsv, lower, upper)
result = cv2.bitwise_and(original,original,mask=mask)
result[mask==0] = (255,255,255)

# Make text black and foreground white
result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
result = cv2.threshold(result, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

cv2.imshow('mask', mask)
cv2.imshow('result', result)
cv2.waitKey()


gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur_image = cv2.medianBlur(image,5)
thresh_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

kernel = np.ones((5,5),np.uint8)
dilate_image = cv2.dilate(image, kernel, iterations = 1)'''

'''import cv2
img = cv2.imread('test1.jpg', 1)
result = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
cv2.imshow("", result)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

