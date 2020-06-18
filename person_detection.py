
import numpy as np
import cv2

class PersonDetection:
    def __init__(self):
        self.setup()
        
    def setup(self):
        self._class = 15 #according to model 15th index is person
        self.color = (0, 255, 0)
        self.conf = 0.2
        
        
        #loading the model
        self.net = cv2.dnn.readNetFromCaffe('./person/MobileNetSSD_deploy.prototxt.txt', './person/MobileNetSSD_deploy.caffemodel')

    def detect(self, frame, drawOnFrame = True):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)
        
        
        self.net.setInput(blob)
        detections = self.net.forward()
        
        
        bboxes = []
        conf = []
    	# loop over the detections
        for i in np.arange(0, detections.shape[2]):
    		# extract the confidence (i.e., probability) associated with
    		# the prediction
            confidence = detections[0, 0, i, 2]
    
    		# filter out weak detections by ensuring the `confidence` is
    		# greater than the minimum confidence
            if confidence > self.conf:
    			# extract the index of the class label from the
    			# `detections`, then compute the (x, y)-coordinates of
    			# the bounding box for the object
                idx = int(detections[0, 0, i, 1])
                
                if not idx == self._class:
                    continue
                
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                bboxes.append(box)
                conf.append(confidence)
                if drawOnFrame:
        			# draw the prediction on the frame
                    label = "{}: {:.2f}%".format("Person",
        				confidence * 100)
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
        				self.color, 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y),
        				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return bboxes, conf
                
