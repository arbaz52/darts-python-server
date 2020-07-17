# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 20:09:49 2020

@author: Shark
"""

import cv2
from deep_sort_pytorch.deep_sort import DeepSort
from pyimagesearch.centroidtracker import CentroidTracker
import numpy as np
import math
#create one for each camera
class Tracking:
    def __init__(self, method=0):
        self.setup(method)
        
        
    def setup(self, method):
        self.method = method
        if method == 0:
            self.deep_sort = DeepSort("./files/ckpt.t7")
        else:
            self.ct = CentroidTracker()
            
        self.color = (255, 0, 0)
        
    def trackUsingDeepSort(self, frame, bboxes, conf, drawOnFrame):
        bboxes_xywh = np.array([[bbox[0], bbox[1], (bbox[2]-bbox[0]), (bbox[3]-bbox[1])*1.4] for bbox in bboxes])
        outputs = self.deep_sort.update(bboxes_xywh, conf, frame)
        if len(outputs) > 0:
            bbox_xyxy = outputs[:,:4]
            identities = outputs[:,-1]
            for i in range(len(outputs)):
                bbox = bbox_xyxy[i]
                w, h = (bbox[2]-bbox[0])/2, (bbox[3]-bbox[1])/2
                bbox += np.array([w,h,w,h], np.int32)
                bboxes[i] = bbox
                if drawOnFrame:
                    _id = identities
                    label = "{}".format(str(_id))
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
        				self.color, 2)
                    y = bbox[1]+ 15
                    cv2.putText(frame, label, (bbox[0], y),
        				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
            return bbox_xyxy, identities
        else:
            return [], []
        
        
        
    def getCentroid(self, bbox):
        cX = int((bbox[0] + bbox[2]) / 2.0)
        cY = int((bbox[1] + bbox[3]) / 2.0)
        return np.array((cX,cY))
    
    def getDist(self, p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    
    def pointInsideRect(self, p, r):
        return p[0] >= r[0] and p[0] <= r[2] and p[1] >= r[1] and p[1] <= r[3]
    
    def trackUsingCT(self, frame, bboxes, drawOnFrame, returnCentroids = False):
        bbs = []
        ids = []
        cs = []
        
        cents = []
        inds = []   #optimizing
        for i, bbox in enumerate(bboxes):
            cents.append(self.getCentroid(bbox))
            inds.append(i)
        #cents = np.array(cents)
        
        objects = self.ct.update(bboxes)
        
    	# loop over the tracked objects
        for (objectID, centroid) in objects.items():
    		# draw both the ID of the object and the centroid of the
    		# object on the output frame
            dist = [self.getDist(c, centroid) for c in cents]
            if(len(dist) <= 0):
                continue
            
            ix = dist.index(min(dist))
            i = inds[ix]
            #opt
            inds.pop(ix)
            _p = cents.pop(ix)
            
            #check if it lies inside the box
            _rect = bboxes[i]
            if not self.pointInsideRect(_p, _rect):
                continue
                
            cs.append(_p)
            
            
            text = "ID {}".format(objectID)
            
            bbs.append(bboxes[i])
            ids.append(objectID)
            if drawOnFrame:
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
        			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                cv2.putText(frame, text, (int(bboxes[i][0]), int(bboxes[i][1])),
        			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        if returnCentroids:
            return bbs, ids, cs
        return bbs, ids
        
        
    '''
        returns a dict of key: id assigned to tracked objects, value: bbox
    '''
    def track(self, frame, bboxes, conf = [], drawOnFrame = True, returnCentroids = False):
        if self.method == 0:
            return self.trackUsingDeepSort(frame, bboxes, conf, drawOnFrame)
        else:
            return self.trackUsingCT(frame, bboxes, drawOnFrame, returnCentroids)
            
            