import numpy as np
import cv2
 

face_cascade = cv2.CascadeClassifier('face_final.xml')
eye_cascade = cv2.CascadeClassifier('eye.xml')
smileCascade = cv2.CascadeClassifier('smile.xml')
 
first_read = True
 

cap = cv2.VideoCapture(0)
ret,img = cap.read()
while(ret):
    ret,img = cap.read()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    gray = cv2.bilateralFilter(gray,5,1,1)
 
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5,minSize=(200,200))
    if(len(faces)>0):
        for (x,y,w,h) in faces:
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
 
            
            roi_face = gray[y:y+h,x:x+w]
            roi_face_clr = img[y:y+h,x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_face,1.3,5,minSize=(50,50))
 
            
            if(len(eyes)>=2):
                
                if(first_read):
                    cv2.putText(img,"Eye detected press (s)",(70,70), cv2.FONT_HERSHEY_PLAIN, 3,(0,255,0),2)
                else:
                    cv2.putText(img,"Eyes open!", (70,70),cv2.FONT_HERSHEY_PLAIN, 2,(255,255,255),2)
            else:
                if(first_read):
                    cv2.putText(img,"No eyes detected", (70,70),cv2.FONT_HERSHEY_PLAIN, 3,(0,0,255),2)
                else:
                    print("Blink detected--------------")
                    cv2.waitKey(3000)
                    first_read=True
        for (x,y,w,h) in faces:
            #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
                
            smile = smileCascade.detectMultiScale(roi_gray,scaleFactor= 1.5,minNeighbors=15,minSize=(25, 25),)
        
            for i in smile:
                if len(smile)>1:
                    cv2.putText(img,"Smiling",(170,450),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3,cv2.LINE_AA)
                else:
                    cv2.putText(img,"Not Smiling",(170,450),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3,cv2.LINE_AA)
             
    else:
        cv2.putText(img,
        "No face detected",(100,100),
        cv2.FONT_HERSHEY_PLAIN, 3,
        (0,255,0),2)
 
    cv2.imshow('img',img)
    a = cv2.waitKey(1)
    if(a==ord('q')):
        break
    elif(a==ord('s') and first_read):
        first_read = False
 
cap.release()
cv2.destroyAllWindows()