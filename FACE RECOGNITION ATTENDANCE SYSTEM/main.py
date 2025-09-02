import cv2
import face_recognition
import numpy as np
import datetime
import os

train='./train'
encoded_names=os.listdir(train)
state=None

encoded_images=[]
for i in encoded_names:
    img=face_recognition.load_image_file(f'{train}/{i}')
    encoded=face_recognition.face_encodings(img)[0]
    encoded_images.append(encoded)
encoded_names_=encoded_names.copy()

cap=cv2.VideoCapture(0)
while True:
    ret, frame=cap.read()
    col=(0,0,255)
    text='AUTHORIZATION DENIED'

    #frame=cv2.resize(frame,(0,0), fx=0.25, fy=0.25)
    frame=frame[:,:,::-1]

    face_location=face_recognition.face_locations(frame)
    if face_location:
        x1,y1,x2,y2=face_location[0]

    encoded=face_recognition.face_encodings(frame, face_location)
    found_names=[]

    for face in encoded:
        match=face_recognition.compare_faces(encoded_images,face)
        face_distance=face_recognition.face_distance(encoded_images,face)
        match_index=np.argmin(face_distance)

        if match[match_index]:
            name=encoded_names[match_index]
            col=(0,255,0)
            text='ENTRY GRANTED'

            found_names.append(name)
            if name in encoded_names:
                if name in encoded_names_:
                    encoded_names_.remove(name)
                if state!=name:
                    with open('log.txt', 'a') as f:  
                        f.write(f'{name[::-1][4:][::-1]} {datetime.datetime.now()}\n')
                        state=name
                        f.close()

        # Display the results
        for (top, right, bottom, left), name in zip(face_location, encoded_names):

            frame = np.ascontiguousarray(frame, dtype=np.uint8)
            cv2.rectangle(frame, (left, top), (right, bottom), col, 4)
            cv2.putText(frame, text, (left, top-13), cv2.FONT_HERSHEY_SIMPLEX, 1, col, 4)





    cv2.imshow('webcam', frame)



    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()