import configparser
import requests
import wget
import ntpath
from os.path import exists
from face import FaceDAndR
import cv2
from Inventory import Suspect, Person, Track
from preprocessing import Preprocessing
import time
import json

import matplotlib.pyplot as plt
import numpy as np

from camera import Camera

import threading

from flask import Flask, Response, make_response


from imutils.object_detection import non_max_suppression

from flask_ngrok import run_with_ngrok

from person_detection import PersonDetection

from logger import Logger

class Server:
    def __init__(self):
        self.log("START", "Server started")
        self.recognizeThresh = 5
        self.xo = 0
        self.lock = threading.Lock()
        

        st = time.time()
        self.log("INFO","Loading Models")
        self.pd = PersonDetection()
        global fdr
        self.fdr = fdr
        
        
        print("Done")
        et = time.time()
        self.log("TIME", "Action took {:2.6f}s".format((et-st)))
        
        self.preprocessings = {}
        
        self.loadConfig()
        data = self.loadServerInfoFromWeb()
        if data is not None:
            self.loadSuspects(data)
            self.loadCameras(data)
            self.loadPreprocessingValuesFromWeb(data)
            
        self.keepProcessingFrames = False
        self.startProcessingFrames()
        self.startWebServer()
        
        
        
    
    
    def startProcessingFrames(self):
        if self.keepProcessingFrames:
            self.log("WARN", "Processing frames - thread already running")
            return True
        self.log("INFO", "Starting processing frames - thread")
        self.keepProcessingFrames = True
        self.thread = threading.Thread(target=self.processFrames, args=()).start()
    
    def stopProcessingFrames(self):
        self.log("INFO", "Stopping processing frames - thread")
        self.keepProcessingFrames = False
        '''
        for k, v in self.cameras.items():
            v.stopThread()
        '''
        
    
        
    
    def processFrameOf(self, camera):
        if not camera.isUp():
            #self.log("WARN", "Video stream for Camera: " + camera._id + " not available")
            return False
        maxt = 10
        frame = None
        for i in range(1, maxt+1):
            #self.log("INFO", "Trying to accesss frame {}/{}".format(i, maxt))
            try:
                ret, f = camera.read()
                if ret:
                    frame = f
            except:
                yyyyy=1
        
        if frame is None:
            #self.log("WARN", "Couldn't access a valid frame")
            return False
        
        
        
        if camera._id in self.preprocessings:
            self.log("INFO", "Pre-Processing frame of camera: "+ camera._id)
            st = time.time()
            lineCoords = [(5, frame.shape[0] - 30*(i+1)) for i in range(3)]
            pp = self.preprocessings[camera._id]
            if 'brightness' in pp:
                bv = pp['brightness']
                frame = Preprocessing.adjustBrightness(frame, bv)
                Preprocessing.putText(frame, "Brightness: " + str(bv), lineCoords[0])
                
            if 'sharpness' in pp:
                sv = pp['sharpness']
                frame = Preprocessing.sharpenImage(frame, k = sv)
                Preprocessing.putText(frame, "Sharpness: " + str(sv), lineCoords[1])
                
            if 'denoise' in pp:
                dv = pp['denoise']
                if dv > 0:
                    frame = Preprocessing.denoiseImage(frame, strength = dv)
                    Preprocessing.putText(frame, "denoise: " + str(dv), lineCoords[2])
            
            et = time.time()
            self.log("TIME", "Action took {:2.6f}s".format((et-st)))
            
        
        
        #person detection
        #plt.imshow(frame)
        self.log("INFO", "Detecting People in the frame")
        bboxes, conf = self.pd.detect(frame, drawOnFrame=False)
        #overlapping bounding boxes
        self.log("INFO", "Applying nms")
        bboxes = non_max_suppression(np.array(bboxes), probs=None, overlapThresh=0.65)
        #tracking
        if len(bboxes) > 0:
            
            tbboxes, tids = camera.tk.track(frame, bboxes, conf, drawOnFrame=False)
            if len(tbboxes) > 0:
                
                self.log("INFO", "Tracking people {}".format(len(tids)))
                for i in range(len(tbboxes)):
                    tbbox = np.array(tbboxes[i], np.int32)
                    tid = tids[i]
                    #increasing fps by selective recognition
                    if camera.track.hasPerson(tid):
                        if camera.track.people[tid].isSuspect():
                            if time.time() - camera.track.people[tid].whenRecognized < self.recognizeThresh:
                                continue
                            
                    person = frame[tbbox[1]:tbbox[3], tbbox[0]:tbbox[2]]
                    #cv2.imshow("person: ", person)
                    faces = fdr.extractFaces(person, drawOnFrame = False)
                    if len(faces) <= 0:
                        continue
                    
                    face = faces[0]
                    fe = fdr.getEmbedding(face[0])
                    
                    #check if he/she is a suspect
                    suspectDetected = False
                    for k, suspect in self.suspects.items():
                        #{"face":face, "em":em, "path":path}  
                        for pic in suspect.pictures:
                            em = pic['em']
                            if fdr.is_match(em, fe):
                                camera.track.suspectDetected(tid, suspect, time.time(), frame, self.SERVER_ID, camera._id)
                                suspectDetected = True
                                break
                        if suspectDetected:
                            break
                    
                #update track
                camera.track.updatePositions(tbboxes, tids)
        
        camera.track.clearForgotten()
        #display bboxes and everything
        camera.track.draw(frame)
        #udpate the processedFrame
        #cv2.imshow("Frame", frame)
        
        t = time.localtime()
        text = "Server: " + time.strftime("%H:%M:%S", t)
        cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1)
        
        with self.lock:
            camera.processedFrame = frame
            camera.processedFrameTime = time.time()
            self.xo = 1
        return True
        
        
    def processFrames(self):
        cameraCurrentlySelected = 0
        keys = list(self.cameras.keys())
        
        self.log("INFO", "Started processing frames")
        while self.keepProcessingFrames:
            key = keys[cameraCurrentlySelected]
            camera = self.cameras[key]
            
            #self.log("INFO", "Processing frame of camera: "+ key)
            st = time.time()
            frameProcessed = self.processFrameOf(camera)
            et = time.time()
            if frameProcessed:
                self.log("TIME", "Processing frame: Action took {:2.6f}s".format((et-st)))
            cameraCurrentlySelected += 1
            
            if cameraCurrentlySelected == len(keys):
                cameraCurrentlySelected = 0
                
            #time.sleep(2)
            
        self.log("INFO", "Stopped processing frames")

        
    def loadConfig(self):
        self.log("INFO", "Loading config")
        confParser = configparser.ConfigParser()
        confParser.read("config.ini")
        
        #get server id
        #information about this server
        self.SERVER_ID = confParser['SERVER']['ID']
        self.SERVER_PORT = confParser['SERVER']['PORT']
        
        
        #web server url
        self.WEB_SERVER = confParser['WEB_SERVER']['URL']
        self.WEB_SERVER_PORT = confParser['WEB_SERVER']['PORT']
        self.log("SUCC", "Done loading")
        
        
            
    def loadServerInfoFromWeb(self):
        self.log("INFO", "fetching info from web!")
        
        #url = "http://"+self.WEB_SERVER+":"+self.WEB_SERVER_PORT+"/server/"
        url = self.WEB_SERVER + "server/"
        print(url)
        
        #request server details
        resp = requests.get(url+self.SERVER_ID)
        if resp.status_code != 200:
            self.log("ERR","Could not fetch server details!")
            return None
        
        data = resp.json()
        if 'err' in data :
            self.log("ERR", data['err']['message'])
            return None
        
        #self.log("DATA", data)
        self.log("SUCC", "info recvd!")
        
        self.data = data
        
        return data

    def loadSuspects(self, data):
        self.log("INFO", "Loading suspects")
        self.suspects = {}
        suspects_data = data['server']['suspects']
        for suspect_data in suspects_data:
            
            tags = suspect_data['tags']
            fullName = suspect_data['fullName']
            _id = suspect_data['_id']
            gender = suspect_data['gender']
            
            #download pictures
            pictures = []
            for picture_url in suspect_data['pictures']:
                #store in suspect_pictures/
                picture_name = ntpath.basename(picture_url)
                
                path = "suspect_pictures/"+picture_name
                if not exists(path):
                    self.log("DOWN", picture_name)
                    wget.download(picture_url, out="suspect_pictures/")
                    self.log("SUCC", path + " downloaded")
                
                #extract face and emb
                img = cv2.imread(path)
                faces = self.fdr.extractFaces(img)
                if len(faces) == 0:
                    print("SKIP", "No face detected")
                    continue
                face = faces[0][0]
                em = self.fdr.getEmbedding(face)
                picture = {"face":face, "em":em, "path":path}  
                pictures.append(picture)
            
            suspect = Suspect(_id, fullName, gender, pictures, tags)
            self.suspects[_id] = suspect
        
        self.log("SUCC", "Suspects loaded")
        return True
        

    def loadCameras(self, data):
        self.log("INFO", "Loading cameras")
        self.cameras = {}
        cameras_data = data['server']['cameras']
        for camera_data in cameras_data:
            _id = camera_data['_id']
            url = camera_data['url']
            lat = camera_data['latitude']
            lng = camera_data['longitude']
            camera = Camera(_id, url, lat, lng)
            self.cameras[_id] = camera
        
            
        self.log("SUCC", "Cameras loaded")
        return True
    
    def loadPreprocessingValuesFromWeb(self, data = None):
        if data is None:
            #loads the preprocessing values for the cameras
            self.log("FETCH", "fetching preprocessing values from web")
            data = self.loadServerInfoFromWeb()
            self.loadPreprocessingValuesFromWeb(data)
        else:
            self.log("UPDATE", "Updating preprocessing values")
            json.dumps(data)
            self.preprocessings = data['server']['preprocessings']
        
    
    '''
    logger
    '''
    def log(self, _type, msg):
        Logger._log(_type, msg)
        
        
        
            
    '''
    web server
    
    '''
    def startWebServer(self):
        self.log("INFO", "Starting web server")
        app = Flask("Python server")
        run_with_ngrok(app)
        self.app = app
        
        @app.route("/logs")
        def logs():
            try:
                s=""
                with open("log.txt", 'r') as fp:
                    for line in fp.readlines():
                        s += line + "<br>"
                    return make_response(s, 200)
            except:
                return make_response("No logs", 200)
        
        @app.route("/start")
        def start():
            self.startProcessingFrames()
            self.log("START", "Starting server")
            return make_response("Starting server", 200)
            
            
        @app.route("/stop")
        def stop():
            self.stopProcessingFrames()
            self.log("STOP", "Stopping server")
            return make_response("Stopping server", 200)
        
        
        @app.route("/updatep")
        def updatePreprocessingValues():
            self.log("UPDATE", "Updating preprocessing values")
            self.loadPreprocessingValuesFromWeb()
            return make_response("Updating preprocessing values", 200)
        
        #server commands to stop processing thread
        @app.route("/startfp")
        def startfp():
            self.startProcessingFrames()
            return make_response("Done!", 200)
            
        @app.route("/stopfp")
        def stopfp():
            self.stopProcessingFrames()
            return make_response("Done!", 200)
        
        @app.route("/camera/<cameraId>")
        def sendProcessedFrame(cameraId):
            with self.lock:
                if self.xo == 1:
                    self.log("HAP", "Value changed")
                    self.xo = 0
                
            if cameraId not in list(self.cameras.keys()):
                self.log("WARN", "Camera doesn't exist")
                return make_response("camera doesn't exist", 404)
            
            #self.log("INFO", "sending processing frame")
            return Response(self.gen(cameraId), mimetype='multipart/x-mixed-replace; boundary=frame')
        
    
        
        
        #self.app.run(host='0.0.0.0', port=self.SERVER_PORT, debug=True,
		#threaded=True, use_reloader=False)
        self.app.run()
        
    def gen(self, cameraId):
        w, h = 300, 200
        frame = None
        lpft = 0
        while True:
            try:
                with self.lock:
                    camera = self.cameras[cameraId]
                    nframe = camera.processedFrame
                    tt = camera.processedFrameTime
                if (tt == lpft):
                    continue
                frame = nframe
                lpft = tt
                #print(frame)
                if frame is None:
                    print("Empty frame")
                    frame = np.random.randint(0,256,(h, w,3), dtype=np.uint8)
            except:
                print("exception while reading frame")
                frame = np.random.randint(0,256,(h, w,3), dtype=np.uint8)
                
            _, encodedImage = cv2.imencode('.jpg', frame)
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                  bytearray(encodedImage) + 
                  b'\r\n')
        

fdr = FaceDAndR() 
server = Server()
cv2.destroyAllWindows()

