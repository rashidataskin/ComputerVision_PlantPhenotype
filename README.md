# ComputerVision_PlantPhenotype

import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv

img = cv2.imread('C:\Python27\Lib\plant_images\100 leaves plant species\data\Acer_Campestre.jpg')
#gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
surf= cv2.SURF(400)
kp, des = surf.detectAndCompute(img, None)
surf.hessianThreshold= 5000
img2 = cv2.drawKeypoints(img,kp,None,(255, 0, 0), 4)
plt.imshow(img2),plt.show()


f = open("resultCamp.txt", "w")
f.write(' , '.join(map(str, kp)))
f.close()
