import cv2
import numpy as np
import time
import requests


class Suspect:
    def __init__(self, _id, fullName, gender, pictures, tags):
        self._id = _id
        self.fullName = fullName
        self.pictures = pictures
        self.gender = gender
        self.tags = tags

class Person:
    def __init__(self, bbox, suspect = None, lastUpdated = time.time()):
        self.bbox = bbox
        self.suspect = suspect
        self.alertGenerated = False
        self.whenRecognized = 0
        self.lastUpdated = lastUpdated
    
    def isSuspect(self):
        return not self.suspect == None
    
    def recognized(self, suspect, t, frame, serverId, cameraId):
        if self.suspect == None:
            self.suspect = suspect
            self.generateAlert(frame, serverId, cameraId)
        elif self.suspect._id == suspect._id:
            print("Same")
        self.whenRecognized = t
        
    def generateAlert(self, frame, serverId, cameraId):
        print("Suspect recognized: " + self.suspect.fullName)
        print("write generate alert code")
        label = "Suspect: " + self.suspect.fullName
        bbox = self.bbox
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)
        
        
        #sending alert to server
        url = "https://darts-web-server.herokuapp.com/server/"+serverId+"/alert/"
        files ={'frame': ('frame.jpg', cv2.imencode(".jpg", frame)[1]) }
        d = {"cameraId": cameraId, "suspectId": self.suspect._id}
        r = requests.post(url,files=files, data=d)
        print(r.json())
        

class Track:
    def __init__(self, updateThresh = 3):
        self.people = {}
        self.updateThresh = 10
    
    def hasPerson(self, tid):
        return tid in list(self.people.keys())
    
    def updatePositions(self, tbboxes, tids):
        for i in range(len(tids)):
            tbbox = tbboxes[i]
            tid = tids[i]
            
            if self.hasPerson(tid):
                self.people[tid].bbox = tbbox
                self.people[tid].lastUpdated = time.time()
            else:
                self.people[tid] = Person(tbbox, lastUpdated=time.time())
        
    def clearForgotten(self):
        x = list(self.people.keys())
        for i in x:
            person = self.people[i]
            if time.time() - person.lastUpdated > self.updateThresh:
                self.people.pop(i)
                print("Removed person")
    
    def suspectDetected(self, tid, suspect, t, frame, serverId, cameraId):
        for k, p in self.people.items():
            if p.suspect == None:
                continue
            elif p.suspect._id == suspect._id:
                if k == tid:
                    p.whenRecognized = t
                return
        
        if self.hasPerson(tid):
            self.people[tid].recognized(suspect, t, frame, serverId, cameraId)
            
    def draw(self, frame):
        #draw bboxes around people
        for i, person in self.people.items():
            _id = i
            bbox = np.array(person.bbox, np.int32)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 3)
            label = "ID: " + str(_id)
            cv2.putText(frame, label, (bbox[0], bbox[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            if person.isSuspect():
                label = "Suspect: " + person.suspect.fullName
                cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)
        #give them ids
        #show suspect name
        
        
        
        