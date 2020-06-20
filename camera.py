import cv2
import numpy as np
from tracking import Tracking
from Inventory import Suspect, Person, Track
import threading

class Camera:
    def __init__(self, _id, url, lat, lng):
        self._id = _id
        self.url = url
        self.lat = lat
        self.lng = lng
        self.processedFrame = None
        self.processedFrameTime = 0
        self.tk = Tracking(1)
        self.track = Track()
        '''
        self.keepThreadRunning = True
        self.lock = threading.Lock()
        self.frame = None
        '''
        
        self.setup()

        
    def setup(self):
        self.cap = cv2.VideoCapture(self.url)
        if self.cap.isOpened():
            print("INFO: Starting the video capture on url: " + self.url)
            #buffer will now hold latest 2 frames
            #self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            #start a thread to read frames and store them in a variable
            '''
            threading.Thread(target=self.get_frame, args=()).start()
            '''
            return True
        else:
            print("ERR: Couldn't start video capture for camera: " + self._id + ", check URL: " + self.url)
            return False
    
    def isUp(self):
        return self.cap.isOpened()
    
    
    def stopThread(self):
        self.keepThreadRunning = False
        
    '''
    def get_frame(self):
        while self.keepThreadRunning:
            ret, frame = self.cap.read()
            if not ret:
                continue
            with self.lock:
                self.frame = frame
        print("STOPPING CAMERA")
    
    def read(self):
        with self.lock:
            return (self.frame is not None), self.frame
    '''
    def read(self):
        if self.isUp():
            ret, frame = self.cap.read()
            
            return ret, frame