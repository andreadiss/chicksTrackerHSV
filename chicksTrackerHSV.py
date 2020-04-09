# import the necessary packages
from collections import deque
import numpy as np
import math
import cv2
import imutils
import time

# create a txt file and start to write cordinates
f = open("coordinates.txt","w+")
f.write("x,y,displacement,speed(pxl/frame)\n")

# initialize the list of tracked points, the frame counter,
# and the coordinate deltas
pts = deque(maxlen=30)
params = deque (maxlen = 30)
counter = 0
(dX, dY) = (0, 0)
direction = ""

# set the color range
lower = (0, 61, 0)
upper = (255, 255, 255)


# read the video file
cap = cv2.VideoCapture("video1.mp4")
 
# allow the camera or video file to warm up
time.sleep(1.0)

while True:
    # grab the current frame
    ret,frame = cap.read()
    if ret == True:

        # resize the frame, blur it and turn into HSV
        frame = imutils.resize(frame, width=600)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
                
        # apply the mask
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None

        # only proceed if at least one contour was found
        if len(cnts) > 0:
                # find the largest contour in the mask, then use
                # it to compute the minimum enclosing circle and
                # centroid
                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
 
                # only proceed if the radius meets a minimum size
                if radius > 10:
                        # draw the circle and centroid on the frame,
                        # then update the list of tracked points
                        cv2.circle(frame, (int(x), int(y)), int(radius),
                                (0, 255, 255), 2)
                        cv2.circle(frame, center, 5, (0, 0, 255), -1)
                        pts.appendleft(center)
                        if counter > 5:
                            centerNew = str(center)
                            centerNew = centerNew.replace("(","").replace(")","").replace(" ","")
                            f.write('{}'.format(centerNew))
        #loop over the set of tracked points
        for i in np.arange(1, len(pts)):
                # if either of the tracked points are None, ignore
                # them
                if pts[i - 1] is None or pts[i] is None:
                        continue
 
                # check to see if enough points have been accumulated in
                # the buffer
                if counter > 5 and i == 1 and pts[-5] is not None:
                        # compute the displacement and speed in pxls/frame
                        
                        displX = int(pts[-5][0] - pts[i][0])
                        displY = int(pts[-5][1] - pts[i][1])            
                        displacement = int(math.sqrt(displX**2+displY**2))
                        speed = int(displacement/5)
                        # display the speed on the frame
                        cv2.putText(frame, "Speed: {} pxl/frame".format(speed),(10, frame.shape[0] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0,0), 1)
                        # write on the txt
                        param = (displacement,speed)
                        params.appendleft(param)
                        paramNew = str(param)
                        paramNew = paramNew.replace("(","").replace(")","").replace(" ","")
                        f.write(',{}\n'.format(paramNew))
                        
                # otherwise, compute the thickness of the line and
                # draw the connecting lines
                thickness = int(np.sqrt(30/float(i+1))*1.5)
                cv2.line(frame, pts[i - 1], pts[i], (114,82,33), thickness)

        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask)       
        counter += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

             
cap.release()
f.close()     
# close all windows
cv2.destroyAllWindows()
