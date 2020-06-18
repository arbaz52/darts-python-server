import cv2
import numpy as np

class Suspect:
    def __init__(self, _id, name, face, em):
        self._id = _id
        self.name = name
        self.face = face
        self.em = em

class Person:
    def __init__(self, bbox, suspect = None):
        self.bbox = bbox
        self.suspect = suspect
        self.alertGenerated = False
    
    def isSuspect(self):
        return not self.suspect == None
    
    def recognized(self, suspect):
        self.suspect = suspect
        self.generateAlert()
        
    def generateAlert(self):
        print("Suspect recognized: " + self.suspect.name)
        print("write generate alert code")
        self.alertGenerated = True

class Track:
    def __init__(self):
        self.people = {}
    
    def hasPerson(self, tid):
        return tid in list(self.people.keys())
    
    def updatePositions(self, tbboxes, tids):
        for i in range(len(tids)):
            tbbox = tbboxes[i]
            tid = tids[i]
            
            if self.hasPerson(tid):
                self.people[tid].bbox = tbbox
            else:
                self.people[tid] = Person(tbbox)
    
    def suspectDetected(self, tid, suspect):
        if self.hasPerson(tid):
            self.people[tid].recognized(suspect)
            
    def draw(self, frame):
        #draw bboxes around people
        for i, person in self.people.items():
            _id = i
            bbox = np.array(person.bbox, np.int32)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 3)
            label = "ID: " + str(_id)
            cv2.putText(frame, label, (bbox[0], bbox[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            if person.isSuspect():
                label = "Suspect: " + person.suspect.name
                cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        #give them ids
        #show suspect name
        
        
        
        