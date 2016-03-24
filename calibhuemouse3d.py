# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 13:16:07 2015

@author: Roshan
"""
#import modules
import numpy as np
import matplotlib
matplotlib.use('tkagg')
import cv2
import os
from LUTptrallr import bgrhsvarray3c
from LUTptrallr import cleanupf
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt

#set camera parameters
os.system('v4l2-ctl -d 0 -c focus_auto=0')
os.system('v4l2-ctl -d 0 -c focus_absolute=0')
os.system('v4l2-ctl -d 0 -c exposure_auto=1')
os.system('v4l2-ctl -d 0 -c exposure_absolute=250')
os.system('v4l2-ctl -d 0 -c contrast=100')
os.system('v4l2-ctl -d 0 -c brightness=100')
os.system('v4l2-ctl -d 0 -c white_balance_temperature_auto=0')
os.system('v4l2-ctl -d 0 -c white_balance_temperature=6500')

#decalre flags
picturestaken = 0 #checks number of pictures calibrated
calibdone = False #status of calibration per picture
calibwindowopen = False #status of calibration window
running = True # status of webcam loop. False = terminate
scene = False #Is the selection window being displayed or video window

#declare calibration parameters
colourfreq = np.zeros((256,256,256))

#region of interest parameters and flags
rect = (0,0,1,1) #coordinates of corners
rectangle = False
rect_over = False
  
def onmouse(event,x,y,flags,params):
    """this function is called on mouse click. it plots colour,saturation and intensity for slected region of interest"""
    # Declare global objects
    global sceneImg,rectangle,rect,ix,iy,rect_over, colourfreq,picturestaken,scene,calibwindowopen
    #Copy img
    sceneCopy = sceneImg.copy()
    # Draw Rectangle
    if event == cv2.EVENT_LBUTTONDOWN:
        rectangle = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if rectangle == True:
            cv2.rectangle(sceneCopy,(ix,iy),(x,y),(255,0,0),1)
            #rect = (min(ix,x),min(iy,y),abs(ix-x),abs(iy-y))
            cv2.imshow('mouse input', sceneCopy)
            cv2.waitKey(1)

    elif event == cv2.EVENT_LBUTTONUP:
        rectangle = False
        rect_over = True
        
	
        #cv2.rectangle(sceneImg,(ix,iy),(x,y),(0,255,0),1)
        # Draw rectangle in copy
        cv2.rectangle(sceneCopy,(ix,iy),(x,y),(255,0,0),1)
        
        
        rect = (min(ix,x),min(iy,y),abs(ix-x),abs(iy-y))       
        filteraxis = np.asarray(bgrhsvarray3(sceneImg,rect[1],rect[1]+rect[3],rect[0],rect[0]+rect[2]))
        colourfreq += filteraxis
        
        #display copy with rectangle for 1 second
        cv2.imshow('mouse input', sceneCopy)
        cv2.waitKey(1000)
        
        picturestaken += 1
        print "%d picturestaken"%(picturestaken)
        scene = False
        cv2.destroyWindow('mouse input')
        calibwindowopen = False
        

# Named window and mouse callback
#cv2.namedWindow('mouse input')
#cv2.setMouseCallback('mouse input',onmouse)
cv2.namedWindow('video')
cap = cv2.VideoCapture(0)
cap.set(cv2.cv.CV_CAP_PROP_FPS, 30)
#cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT,720)
#cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH,920)
keyPressed = None

#print instructions
print "Press s to select region. *Warning: do not press s if mouse input window not present"
# Start video stream
while running:
    readOK, frame = cap.read()
    calibdone = False	
    keyPressed = cv2.waitKey(5)
    if keyPressed == ord('s'):
    	
    	#while calibdone == False:
        scene = True
        cv2.destroyWindow('video')
        print "Select object of interest"
        sceneImg = frame.copy() 
        cv2.imshow('mouse input', sceneImg)
	
    if not calibwindowopen:
    	cv2.namedWindow('mouse input')
    	cv2.setMouseCallback('mouse input',onmouse)
    	calibwindowopen = True
    if not scene:
        cv2.imshow('video', frame)
    if picturestaken == 2:
    	running = False
    
fig = plt.figure(figsize = (8,6))
ax = fig.add_subplot(111,projection = '3d')
#sortedh = np.sort(hueaxis)
#huevals = np.array(np.where(hueaxis >= sortedh[-3])).tolist() #list with double bracket
#print 'hue values of ball =  %s. Please refer to plot' %(', '.join(str(it) for it in huevals[0]))

interestpos = np.nonzero(colourfreq)
xs = interestpos[0]
ys = interestpos[1]
zs = interestpos[2]
colval = colourfreq[np.nonzero(colourfreq)]

colors = cm.brg(colval/max(colval))
colmap = cm.ScalarMappable(cmap = cm.brg)
colmap.set_array(colval/max(colval))

yg = ax.scatter(xs,ys,zs, c=colors, marker='s')
cb = fig.colorbar(colmap)

ax.set_xlabel('Hue')
ax.set_ylabel('Saturation')
ax.set_zlabel('Value')
#plt.ion()
cleanupf()
cv2.destroyAllWindows()
cap.release()
plt.show()