import cv2
import numpy as np
from tracking import Tracking
from Inventory import Suspect, Person, Track
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
        self.track = Track()
        self.setup()

        
    def setup(self):
        self._stream = None
        self.bytes = bytes()
        try:
            Logger._log("INFO", "Opening camera stream!")
            self._stream = urllib.request.urlopen('http://localhost:8082/')
            Logger._log("SUCC", "Camera stream opened!")
        except:
            Logger._log("WARN", "Couldn't start stream - Camera stream unavailable")
            
        
    def isUp(self):
        return self._stream is not None
    
    def read(self):
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