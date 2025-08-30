
import cv2
from pyzbar.pyzbar import decode
import matplotlib.pyplot as plt
import numpy as np
import datetime

users=[]
logged=[]
with open('users.txt', 'r') as f:
    users=[l[:-1] for l in f.readlines() if len(l)>2]
f.close()

cap=cv2.VideoCapture(0)
while True:
    ret, frame=cap.read()

    qr_info=decode(frame)
    

    if len(qr_info):
        qr_info=qr_info[0]
        rect=qr_info.rect
        polygon=qr_info.polygon
        secret=qr_info.data

        if secret.decode() in users:
            cv2.putText(frame, 'ENTRY GRANTED', (rect.left, rect.top-13), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
            if secret.decode() not in logged:
                with open('log.txt', 'a') as f:  
                    f.write(f'{secret.decode()} {datetime.datetime.now()}\n')
                    logged.append(secret.decode())
                    f.close()

        else:
            cv2.putText(frame, 'AUTHORIZATION DENIED', (rect.left, rect.top-13), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)


        frame=cv2.rectangle(frame, (rect.left, rect.top), (rect.left+rect.width, rect.top+rect.height), (0, 0, 255), 4)
        frame=cv2.polylines(frame, [np.array(polygon)], True, (0, 255,0), 4)
    
    cv2.imshow('webcam', frame)



    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()