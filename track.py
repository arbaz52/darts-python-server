import cv2
import numpy as np
import time

faceMatchThresh = 5 #seconds for now

class Suspect:
    def __init__(self, _id, name, face, em):
        self._id = _id
        self.name = name
        self.face = face
        self.em = em
        self.person = None
        self.whenRecognized = 0
        
    def recognized(self, person):
        if self.person == None:
            self.person = person
            self.generateAlert()
        elif self.person._id == person._id:
            print("Same")
        self.whenRecognized = time.time()
        self.generateAlert()
        
    
    def match(self, person, fdr):
        global faceMatchThresh
        if self.person == None or time.time() - self.whenRecognized > faceMatchThresh:
            if fdr.is_match(self.em, person.em):
                self.recognized(person)
            
    
    def isBeingTracked(self):
        global faceMatchThresh
        return (not self.person == None) and (time.time() - self.whenRecognized > faceMatchThresh)

class Person:
    def __init__(self, _id, bbox, face = None, lastUpdated = time.time()):
        self.bbox = bbox
        self.lastUpdated = lastUpdated
    
    def isSuspect(self):
        return not self.suspect == None
    
    def recognized(self, suspect, t):
        if self.suspect == None:
            self.suspect = suspect
            self.generateAlert()
        elif self.suspect._id == suspect._id:
            print("Same")
        self.whenRecognized = t
        self.suspect = suspect
        self.generateAlert()
        
    def generateAlert(self):
        print("Suspect recognized: " + self.suspect.name)
        print("write generate alert code")
        self.alertGenerated = True

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
    
    def suspectDetected(self, tid, suspect, t):
        if self.hasPerson(tid):
            self.people[tid].recognized(suspect, t)
            
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
        
        
        
        