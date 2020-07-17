import time
import cv2
import requests

class Suspect:
    def __init__(self, _id, fullName, gender, tags, pictures, fdr):
        self._id = _id
        self.gender = gender
        self.fullName = fullName
        self.tags = tags
        self.pictures = pictures
        self.personId = None
        self.last_time_recognized = None
        self.last_time_alert_generated = None
        self.loadEmbeddings(fdr)
    
    def loadEmbeddings(self, fdr):
        self.embds = []
        print("loading embeddings")
        for pic in self.pictures:
            faces = fdr.extractFaces(cv2.imread(pic))
            if len(faces) <= 0:
                continue
            face = fdr.getEmbedding(faces[0][0])
            self.embds.append(face)
        
    def recognized(self, personId, frame, serverId, cameraId):
        self.last_time_recognized = time.time()
        if personId == self.personId:
            print("Already recognized " + self.fullName)
        else:
            print("Alert for: " + self.fullName)
            self.personId = personId
            self.generateAlert(frame, serverId, cameraId)

    def shouldRecognize(self):
        return self.last_time_recognized is None or time.time() - self.last_time_recognized > 10 
    
    
    def generateAlert(self, frame, serverId, cameraId):
        if self.last_time_alert_generated is None or time.time() - self.last_time_alert_generated > 20:
            self.last_time_alert_generated = time.time()
            print("Generating alert for: " + self.fullName)
            #sending alert to server
            url = "https://darts-web-server.herokuapp.com/server/"+serverId+"/alert/"
            files ={'frame': ('frame.jpg', cv2.imencode(".jpg", frame)[1]) }
            d = {"cameraId": cameraId, "suspectId": self._id}
            r = requests.post(url,files=files, data=d)
            print(r.json())
        else:
            print("Alert generated for " + self.fullName + "less than 20 seconds ago!")
    
    
    def update(self):
        if self.shouldRecognize():
            print(self.fullName + " up for recognition")
    
class Inventory:
    def __init__(self, suspects):
        self.suspects = suspects
    
    def update(self):
        for suspect in self.suspects:
            suspect.update()
    
    
        
    
    