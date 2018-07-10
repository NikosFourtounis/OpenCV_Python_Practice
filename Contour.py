import numpy as np
import cv2 as cv

img = cv.imread('download.jpg')
blurred = cv.pyrMeanShiftFiltering(img,31,91)
gray = cv.cvtColor(blurred,cv.COLOR_BGR2GRAY)
ret,thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)

_,contours,_ = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

cv.drawContours(img, contours, -1, (0,0,255), 6)

cv.namedWindow('Display', cv.WINDOW_NORMAL)
cv.imshow('Display', img)
cv.waitKey()