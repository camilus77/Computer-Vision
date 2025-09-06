import ultralytics
import math

import cv2
import pandas as pd
from ultralytics import YOLO

model=YOLO('yolo11n.pt')
class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane','bus', 'train','truck']

#this function
def tracker(objects_rect,center_points={}, id_count = 0):
    # Objects boxes and ids
    objects_bbs_ids = []

    # Get center point of new object
    for rect in objects_rect:
        x, y, w, h = rect
        cx = (x + x + w) // 2
        cy = (y + y + h) // 2

        # Find out if that object was detected already
        same_object_detected = False
        for id, pt in center_points.items():
            dist = math.hypot(cx - pt[0], cy - pt[1])

            if dist < 35:
                center_points[id] = (cx, cy)
#                   print(self.center_points)
                objects_bbs_ids.append([x, y, w, h, id])
                same_object_detected = True
                break

        # New object is detected we assign the ID to that object
        if same_object_detected is False:
            center_points[id_count] = (cx, cy)
            objects_bbs_ids.append([x, y, w, h, id_count])
            id_count += 1

    # Clean the dictionary by center points to remove IDS not used anymore
    new_center_points = {}
    for obj_bb_id in objects_bbs_ids:
        _, _, _, _, object_id = obj_bb_id
        center = center_points[object_id]
        new_center_points[object_id] = center

    # Update dictionary with IDs not used removed
    center_points = new_center_points.copy()
    return objects_bbs_ids

#tracker=Tracker()
count=0
down={}
up={}

counter_down=set()
counter_up=set()

cap=cv2.VideoCapture('traffic2.mp4')

while True:
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    frame=cv2.resize(frame,(1020,500))

    #get prediction for each frame. What and what did it see here?
    results=model.predict(frame)

    a=results[0].boxes.data
    a = a.detach().cpu().numpy()
    px=pd.DataFrame(a).astype("float")

    list=[]
    for row in range (len(px)):
        x1,y1,x2,y2,_,cl_id=px.iloc[row,:].values.flatten().tolist()
        c=class_list[int(cl_id)]
        if c=='car':
            list.append([x1,y1,x2,y2])

    bbox_id=tracker(list)#.update(list)
    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        #cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
        #cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)

        red_line_y=300
        blue_line_y=400
        offset = 4
        h1=0



        ''' both lines combined condition . First condition is for red line'''
        ## condition for counting the cars which are entering from red line and exiting from blue line
        if red_line_y < (cy + offset) and red_line_y > (cy - offset):
            down[id]=cy
        
        if id in down:
           if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
             #cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
             #cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),24)
             #counter+=1
             counter_down.add(id)  # get a list of the cars and buses which are entering the line red and exiting the line blue

        # condition for cars entering from  blue line
        if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
          up[id]=cy

        if id in up:
           if red_line_y > (cy - offset):

             #cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
             #cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
             #if id
             counter_up.add(id)  # get a list of the cars which are entering the line 1 and exiting the line 2

    text_color = (255,255,255)
    red_color = (0, 0, 255)  # (B, G, R)
    blue_color = (255, 0, 0)  # (B, G, R)
    green_color = (0, 255, 0)  # (B, G, R)

    #cv2.line(frame,(172,300),(774,300),red_color,3)  #  starting cordinates and end of line cordinates
    #cv2.putText(frame,('red line'),(172,198),cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    #cv2.line(frame,(8,400),(927,400),blue_color,3)  # seconde line
    #cv2.putText(frame,('blue line'),(8,268),cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    downwards = (len(counter_down))
    cv2.putText(frame,('going down - ')+ str(downwards),(60,40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, green_color, 1, cv2.LINE_AA)


    upwards = (len(counter_up))
    cv2.putText(frame,('going up - ')+ str(upwards),(60,60),cv2.FONT_HERSHEY_SIMPLEX, 0.5, red_color, 1, cv2.LINE_AA)


    #cv2_imshow(frame)
    cv2.imshow("frames", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()



