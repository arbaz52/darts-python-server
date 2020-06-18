import cv2
from person_detection import PersonDetection
from tracking import Tracking
import numpy as np

cap = cv2.VideoCapture("./files/output_0.avi")
mode = 1    #simple just video
w, h = int(cap.get(3)), int(cap.get(4))
forcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("./files/output_"+str(mode)+".avi", forcc, 25, (w, h))

pd = PersonDetection()
tk = Tracking()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    
    if mode == 1:
        #person detection
        bboxes, conf = pd.detect(frame)
        #tracking
        if len(bboxes) > 0:
            tracks = tk.track(frame, bboxes, conf)
            
    out.write(frame)
    print(".", end="")
    cv2.imshow("frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
out.release()
cv2.destroyAllWindows()