import cv2
import numpy as np

image = cv2.imread('4.jpg')
#image = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)

kernel = np.ones((7,7),np.uint8)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_bound = np.array([90, 50, 70])
upper_bound = np.array([128, 255, 255])

mask = cv2.inRange(hsv, lower_bound, upper_bound)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
segmented_image = cv2.bitwise_and(image, image, mask=mask)

contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output = cv2.drawContours(image, contours, -1, (0, 0, 255), 3)

for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    crop_image = output[y:y+h,x:x+w]
    cv2.imshow("image", crop_image)
    cv2.waitKey(0)

result = cv2.drawContours(image, contours, contourIdx=-1, color=(255,255,255),thickness=-1)
cv2.imshow("result", result)
cv2.waitKey(0)

