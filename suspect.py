from video_analyser import VideoAnalyser
import cv2

import os
import wget

class Suspect:
    def __init__(self, _id, fullName, gender, pictures):
        self._id = _id;
        self.fullName = fullName
        self.gender = gender
        self.pictures = pictures
        
    def downloadPicture(self, url):
        filename = os.path.basename(url)
        path = "tmp/"+filename
        if(os.path.exists(path)):
            return path
        wget.download(url, path)
        return path
        
    def generateEmbeddings(self, va: VideoAnalyser):
        self.embeddings = []
        for pic in self.pictures:
            path = self.downloadPicture(pic)
            faces = va.extractFaces(cv2.imread(path))
            if len(faces) > 0:
                self.embeddings.append(va.getEmbedding(faces[0][0]))
            else:
                print("INFO: face not found in provided picture of suspect")