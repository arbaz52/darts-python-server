from mtcnn import MTCNN
from PIL import Image
import numpy as np
import cv2
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from scipy.spatial.distance import cosine
import math

class FaceDAndR:
    def __init__(self):
        self.detector = MTCNN()
        self.vggface = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
        
    def getCentroid(self, bbox):
        cX = int((bbox[0] + bbox[2]) / 2.0)
        cY = int((bbox[1] + bbox[3]) / 2.0)
        return np.array((cX,cY))
    
    def getDist(self, p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    
    def rectInsideRect(self, r1, r2, thresh = 32):
        return r1[0]+thresh > r2[0] and r1[2]-thresh < r2[2] and r1[1]+thresh > r2[1] and r1[3]-thresh < r2[3]
        
    def extractFacesAndAssignToPeople(self, frame, bboxes, ids, cents, requiredSize = (224, 224), drawOnFrame=True):
        results = self.detector.detect_faces(frame)
        bboxes = bboxes.copy() #poping from these to improve performance
        ids = ids.copy() #poping
        cents = cents.copy() #poping
        
        faces = []
        for result in results:
            x1, y1, width, height = result['box']
            x2, y2 = x1 + width, y1 + height
            face = frame[y1:y2, x1:x2]
            #image = Image.fromarray(face)
            
            #calculate distance
            #find the closest
            #check if it fits
            #if it fits, remove and assign this face id of the person
            centroid = self.getCentroid([x1,y1,x2,y2])
            dist = [self.getDist(c, centroid) for c in cents]
            if(len(dist) <= 0):
                continue
            
            ix = dist.index(min(dist))
            _rect = bboxes[ix]
            if not self.rectInsideRect([x1,y1,x2,y2], _rect):
                continue
            
            
            try:
                image = cv2.resize(face, requiredSize)
                _id = ids[ix]
                face_array = np.asarray(image)
                #adding embds too
                embd = self.getEmbedding(face_array)
                
                
                faces.append((face_array, (x1, y1, x2, y2), _id, embd))
                ids.pop(ix)
                cents.pop(ix)
                bboxes.pop(ix)
            except:
                print("face size isn't fit for the model")
            
            
            #draw
            if drawOnFrame:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
                text="ID:"+str(_id)
                cv2.putText(frame, text, (x1, y1),
        			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            #image = image.resize(requiredSize)
            
        return faces
    
    
    def extractFaces(self, frame, requiredSize = (224, 224), drawOnFrame=True):
        results = self.detector.detect_faces(frame)
        faces = []
        for result in results:
            x1, y1, width, height = result['box']
            x2, y2 = x1 + width, y1 + height
            face = frame[y1:y2, x1:x2]
            #image = Image.fromarray(face)
            
            #draw
            if drawOnFrame:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
            
            #image = image.resize(requiredSize)
            try:
                image = cv2.resize(face, requiredSize)
                face_array = np.asarray(image)
                faces.append((face_array, (x1, y1, x2, y2)))
            except:
                print("face size isn't fit for the model")
            
        return faces
        
    def getEmbedding(self, face):
        samples = np.expand_dims(face, 0)
        samples = (samples.astype('float64'))
        # prepare the face for the model, e.g. center pixels
        samples = preprocess_input(samples, version=2)
        yhat = self.vggface.predict(samples)
        return yhat
    
    def is_match(self, emb1, emb2, thresh = 0.39):
    	score = cosine(emb1, emb2)
    	#print(score)
    	if score <= thresh:
            #print('>face is a Match (%.3f <= %.3f)' % (score, thresh))
            return True
    	else:
            #print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh))
            return False