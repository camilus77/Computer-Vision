import ultralytics
import math
import cv2
import pandas as pd
from ultralytics import YOLO
import time

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

detected=set()

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

    bbox_id=tracker(list)
    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2

        red_line_y=200
        blue_line_y=250
        offset = 4
        h1=0



        ''' both lines combined condition . First condition is for red line'''
        ## condition for counting the cars which are entering from red line and exiting from blue line
        if red_line_y < (cy + offset) and red_line_y > (cy - offset):
            down[id]=time.time()
        
        if id in down:
            if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
                time1=down[id]
                time2=time.time()
                if time1==time2:
                    speed=0
                else:
                    speed=50/(time2-time1)
                detected.add(id)

                cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                cv2.rectangle(frame,(int(x3),int(y3)),(int(x4),int(y4)),(0,0,255),2)
                cv2.putText(frame,('Speed:'+ str(round(speed*3.6,2))+'km/hr'),(int(x4),int(y3)),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,255),1)
                cv2.putText(frame,('Cars Detected - ')+ str(len(detected)),(60,60),cv2.FONT_HERSHEY_SIMPLEX, 0.5, red_color, 1, cv2.LINE_AA)
    
                #counter+=1

        # condition for cars entering from  blue line
        if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
            up[id]=time.time()

        if id in up:
            if red_line_y > (cy - offset):
                time1=up[id]
                time2=time.time()
                if time1==time2:
                    speed=0
                else:
                    speed=50/(time2-time1)
                detected.add(id)

                cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                cv2.rectangle(frame,(int(x3),int(y3)),(int(x4),int(y4)),(0,0,255),2)
                cv2.putText(frame,('Speed:'+ str(round(speed*3.6,2))+'km/hr'),(int(x4),int(y3)),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,255),1)
    
    text_color = (255,255,255)
    red_color = (0, 0, 255)  # (B, G, R)
    blue_color = (255, 0, 0)  # (B, G, R)
    green_color = (0, 255, 0)  # (B, G, R)


    #cv2_imshow(frame)
    cv2.imshow("frames", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()



