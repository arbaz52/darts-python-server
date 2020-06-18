import cv2
import numpy as np


class Camera:
    def __init__(self, _id, url, lat, lng):
        self._id = _id
        self.url = url
        self.lat = lat
        self.lng = lng
        self.setup()

        
    def setup(self):
        self.cap = cv2.VideoCapture(self.url)
        if self.cap.isOpened():
            print("INFO: Starting the video capture on url: " + self.url)
            self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            print("ERR: Couldn't start video capture for camera: " + self._id + ", check URL")
            print("SKIP: " + self._id)
            self.w = 300
            self.h = 200
    
    def get_frame(self):
        try:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    #return None
                    print("WARN: Empty frame")
                    frame = np.random.randint(0,256,(self.h, self.w,3), dtype=np.uint8)
            else:
                #return None
                print("ERR: Connection unavailable for camera: " + self._id)
                frame = np.random.randint(0,256,(self.h, self.w,3), dtype=np.uint8)
                
        except:
            #return None
            print("ERR: exception while reading frame")
            frame = np.random.randint(0,256,(self.h, self.w,3), dtype=np.uint8)
        
        return frame