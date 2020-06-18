import cv2
from person_detection import PersonDetection
from tracking import Tracking
from imutils.video import FPS
from face import FaceDAndR
import numpy as np

from Inventory import Suspect, Person, Track

import time

cap = cv2.VideoCapture(0)

st = time.time()
print("Loading Models")
pd = PersonDetection()
tk = Tracking(1)
#fdr = FaceDAndR()
print("Done")
et = time.time()
print("Time: " + str(et-st) + "s")

st = time.time()
print("Loading suspects")
files = ["./tmp/arbaz.png", "./tmp/majid.png"]
suspects = []
for f in files:
    img = cv2.imread(f)
    faces = fdr.extractFaces(img)
    if len(faces) <= 0:
        continue
    
    face = faces[0]
    emb = fdr.getEmbedding(face[0])
    suspects.append(Suspect(f, f, face, emb))


print("Done")
et = time.time()
print("Time: " + str(et-st) + "s")



print("Setting up inventory")
track = Track()
print("Done")




fps = FPS().start()
i = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    #person detection
    bboxes, conf = pd.detect(frame, drawOnFrame=False)
    #tracking
    if len(bboxes) > 0:
        tbboxes, tids = tk.track(frame, bboxes, conf, drawOnFrame=False)
        if len(tbboxes) > 0:
            for i in range(len(tbboxes)):
                tbbox = np.array(tbboxes[i], np.int32)
                tid = tids[i]
                person = frame[tbbox[1]:tbbox[3], tbbox[0]:tbbox[2]]
                #cv2.imshow("person: ", person)
                faces = fdr.extractFaces(person, drawOnFrame = False)
                if len(faces) <= 0:
                    continue
                
                face = faces[0]
                fe = fdr.getEmbedding(face[0])
                
                #check if he/she is a suspect
                for suspect in suspects:
                    if fdr.is_match(suspect.em, fe):
                        track.suspectDetected(tid, suspect)
                        #label = "{}: {}".format("Suspect: ", suspect.name)
                        #cv2.putText(frame, label, (tbbox[0], tbbox[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            #update track
            track.updatePositions(tbboxes, tids)
    
    #display bboxes and everything
    track.draw(frame)
    
    cv2.imshow("frame", frame)
    
    fps.update()
    i += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
fps.stop()
print("AVG FPS: " + str(fps.fps()))


cap.release()
cv2.destroyAllWindows()