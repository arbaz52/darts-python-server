from mtcnn import MTCNN
from PIL import Image
import numpy as np
import cv2
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from scipy.spatial.distance import cosine

class FaceDAndR:
    def __init__(self):
        self.detector = MTCNN()
        self.vggface = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
        
        
    
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
            image = cv2.resize(face, requiredSize)
            face_array = np.asarray(image)
            faces.append((face_array, (x1, y1, x2, y2)))
            
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