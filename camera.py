import cv2
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
            
        else:
            print("ERR: Couldn't start video capture for camera: " + self._id + ", check URL")
            print("SKIPPING: " + self._id)