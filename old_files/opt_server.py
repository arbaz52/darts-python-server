##no person tracking
#loop
    #based on threshold (seconds after which the suspect should be looked for again) update up-for-recognition-list
    
    #detect faces
    #if faces found
    #start face recog for each against suspects in up-for-recognition
    #if face recog
    #send alert, add to alert-generated-for-this-suspect
    
#selectuve tracking
#loop
    #threshold-based up-for-recognition
    #detect faces face-coords
    #detect people
    #join faces-with-people
    #each camera will have its own tracker
    #an inventory, same for all the cameras
    #suspects will have the person being tracked
    

import cv2

def log(s):
    print(s)

def textOnFrame(frame, label, org, fc=(255,255,255), bc=(0,0,0)):
    x1 = org[0]
    y1 = org[1]
    img = frame
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    cv2.rectangle(img,(x1, y1-5),(x1+t_size[0]+3,y1+t_size[1]+4+5), bc,-1)
    cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 1, fc, 1)

from person_detection import PersonDetection
from tracking import Tracking
from face import FaceDAndR
from opt_inventory import Inventory, Suspect
import time

log("loading models")
#fdr = FaceDAndR()
pd = PersonDetection()
tk = Tracking(1)
log("done")
log("loading inventory")
cameraId = "camer"
suspects = []
suspects.append(Suspect("1", "affan", ["hacker"], ["suspect_pictures/affan.png"], fdr))
suspects.append(Suspect("2", "osama", ["hacker"], ["suspect_pictures/osama.png"], fdr))
inventory = Inventory(suspects)

cap = None
noframescount = 0

def connect():
    global cap, noframescount
    noframescount = 0
    log("connecting to camera")
    cap = cv2.VideoCapture("http://localhost:8082")
    log("done")
    
connect()

while cap != None and cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        noframescount += 1
        if noframescount > 30:
            connect()
        continue
    
    noframescount = 0
    
    t = 0
    
    st = time.time()
    bboxes, conf = pd.detect(frame, drawOnFrame=False)
    et = time.time()
    t += (et-st)
    #print("detection time taken: {:2.4f}s".format(et-st))
    
    st = time.time()
    bboxes, ids, cents = tk.track(frame, bboxes, conf, returnCentroids=True, drawOnFrame=False)
    et = time.time()
    t += (et - st)
    #print("tracking time taken: {:2.4f}s".format(et-st) )
    
    st = time.time()
    facesWithIds = fdr.extractFacesAndAssignToPeople(frame, bboxes, ids, cents, drawOnFrame=False)
    et = time.time()
    t += (et - st)
    #print("extracting faces time taken: {:2.4f}s".format(et-st) )
    
    #drawing normal boxes around detected people
    for i in range(len(bboxes)):
        _bbox = bboxes[i]
        _id = ids[i]
        _cent = cents[i]
        st = (int(_bbox[0]), int(_bbox[1]))
        end = (int(_bbox[2]), int(_bbox[3]))
        clr = (0, 255, 0)
        cv2.rectangle(frame, st, end, clr, 2)
        label="ID:{}".format(_id)
        textOnFrame(frame, label, fc=(0,0,0), bc=(0,255,0), org=(int(_cent[0]), int(_cent[1])))
    
    for fd in facesWithIds:
        _c = fd[1]
        st = (_c[0], _c[1])
        end = (_c[2], _c[3])
        cv2.rectangle(frame, st, end, (255, 0, 0), 2)
    
    inventory.update()
    for suspect in inventory.suspects:
        #recognition
        if suspect.shouldRecognize():
            recognized = -1
            for embd in suspect.embds:
                for index, facedata in enumerate(facesWithIds):
                    if fdr.is_match(facedata[3], embd):
                        recognized = index
                        break
                if recognized >= 0:
                    faceWithId = facesWithIds.pop(recognized)
                    personId = cameraId + "_" + str(faceWithId[2])
                    suspect.recognized(personId)
                    break
                
            if recognized == -1:
                if suspect.last_time_recognized != None:
                    if time.time() - suspect.last_time_recognized > 20:
                        suspect.personId = None
                
        #displaying the person red on frame
        if suspect.personId != None:
            _trackId = int(suspect.personId.split("_")[1])
            try:
                _trackIdIndex = ids.index(_trackId)
                if not _trackIdIndex > -1:
                    continue
            except:
                continue
                
            _bbox = bboxes[_trackIdIndex]
            x1,y1,x2,y2 = _bbox[0],_bbox[1],_bbox[2],_bbox[3]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
            text = "Suspect: " + suspect.name
            textOnFrame(frame, text, (int(x1), int(y1+5)), bc=(0, 0, 255), fc=(0,0,0))
            
    
    
    print("processing one frame: {:2.4f}s".format(t))
    
    
    cv2.imshow("frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()