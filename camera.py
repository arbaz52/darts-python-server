import cv2
import numpy as np
from tracking import Tracking
#from Inventory import Suspect, Person, Track
import urllib.request
from logger import Logger

class Camera:
    def __init__(self, _id, url, lat, lng):
        self._id = _id
        self.url = url
        self.lat = lat
        self.lng = lng
        self.processedFrame = None
        self.processedFrameTime = 0
        self.tk = Tracking(1)
        #self.track = Track()
        self.setup()
        self.invalidframescount = 0
        
    
    def reconnect(self):
        self.invalidframescount += 1
        if self.invalidframescount > 30:
            self.connect()
    
    def connect(self):
        self.invalidframescount = 0
        Logger._log("CONN", "connecting to camera " + self._id + ", url: " + self.url)
        self.cap = cv2.VideoCapture(self.url)
        Logger._log("DONE", "done")

        
    def setup(self):
        self.connect()
    
    def _setupUsingVC(self):
        try:
            Logger._log("INFO", "Opening video capture for camera " + self._id)
            self.cap = cv2.VideoCapture(self.url)
            Logger._log("SUCC", "Video Capture opened!")
        except:
            Logger._log("ERR", "Couldn't open video capture for camera " + self._id)
        
    def _setupUsingStream(self):
        self._stream = None
        self.bytes = bytes()
        try:
            Logger._log("INFO", "Opening camera stream!")
            self._stream = urllib.request.urlopen(self.url)
            Logger._log("SUCC", "Camera stream opened!")
        except:
            Logger._log("WARN", "Couldn't start stream - Camera stream unavailable")
            
        
    def isUp(self):
        return self.cap.isOpened()
    
    def read(self):
        return self._readVC()
        
    def _readVC(self):
        if self.isUp():
            return self.cap.read()
        return False, None
        
    def readStream(self):
        if self.isUp():
            self.bytes += self._stream.read(1024)
            a = self.bytes.find(b'\xff\xd8')
            b = self.bytes.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = self.bytes[a:b+2]
                self.bytes = self.bytes[b+2:]
                i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                return True, i
        return False, None