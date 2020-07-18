import configparser
import requests
import wget
import ntpath
from os.path import exists
from face import FaceDAndR
import cv2
from opt_inventory import Suspect, Inventory
from preprocessing import Preprocessing
import time
import json
from jsoner import Jsoner
import numpy as np
import os
import sys
from camera import Camera

import threading

from flask import Flask, Response, make_response


from imutils.object_detection import non_max_suppression

from flask_ngrok import run_with_ngrok

from person_detection import PersonDetection

from logger import Logger

class Server:
    CONFIG_FILE_PATH = "config.json"
    def __init__(self):
        self.log("START", "Server started", True)
        self.xo = 0
        self.lock = threading.Lock()


        st = time.time()
        self.log("INFO","Loading Models", True)
        self.pd = PersonDetection()
        global fdr
        self.fdr = fdr
        
        
        et = time.time()
        self.log("TIME", "Action took {:2.6f}s".format((et-st)), True)
        
        self.preprocessings = {}
        
        self._loadConfig()
        data = self.loadServerInfoFromWeb()
        if data is not None:
            self.loadSuspects(data)
            self.loadCameras(data)
            self.loadPreprocessingValuesFromWeb(data)
            self.keepProcessingFrames = False
            self.startProcessingFrames()
            self.startWebServer()
        else:
            self.log("ERR", "Couldn't load information from (REST API), check internet connection and URL", True)
            return
        
        
    def textOnFrame(self, frame, label, org, fc=(255,255,255), bc=(0,0,0)):
        x1 = org[0]
        y1 = org[1]
        img = frame
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        cv2.rectangle(img,(x1, y1-5),(x1+t_size[0]+3,y1+t_size[1]+4+5), bc,-1)
        cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 1, fc, 1)

        
    
    
    def startProcessingFrames(self):
        if self.keepProcessingFrames:
            self.log("WARN", "Processing frames - thread already running", True)
            return True
        self.log("INFO", "Starting processing frames - thread", True)
        self.keepProcessingFrames = True
        self.thread = threading.Thread(target=self.processFrames, args=()).start()
    
    def stopProcessingFrames(self):
        self.log("INFO", "Stopping processing frames - thread", True)
        self.keepProcessingFrames = False
        
    
        
    
    def processFrameOf(self, camera):
        if not camera.isUp():
            #self.log("WARN", "Video stream for Camera: " + camera._id + " not available")
            camera.connect()
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
            camera.reconnect()
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
            
        
        
        #processing
        cameraId = camera._id
        t = 0
        
        st = time.time()
        bboxes, conf = self.pd.detect(frame, drawOnFrame=False)
        et = time.time()
        t += (et-st)
        if len(bboxes) == 0:
            return False
        #print("detection time taken: {:2.4f}s".format(et-st))
        
        st = time.time()
        bboxes, ids, cents = camera.tk.track(frame, bboxes, conf, returnCentroids=True, drawOnFrame=False)
        et = time.time()
        t += (et - st)
        if len(bboxes) == 0:
            return False
        #print("tracking time taken: {:2.4f}s".format(et-st) )
        
        st = time.time()
        facesWithIds = fdr.extractFacesAndAssignToPeople(frame, bboxes, ids, cents, drawOnFrame=False)
        et = time.time()
        t += (et - st)
        #print("extracting faces time taken: {:2.4f}s".format(et-st) )
        
        #drawing normal boxes around detected people
        for i in range(len(bboxes)):
            _bbox = bboxes[i]
            _id = ids[i]
            _cent = cents[i]
            st = (int(_bbox[0]), int(_bbox[1]))
            end = (int(_bbox[2]), int(_bbox[3]))
            clr = (0, 255, 0)
            cv2.rectangle(frame, st, end, clr, 2)
            label="ID:{}".format(_id)
            self.textOnFrame(frame, label, fc=(0,0,0), bc=(0,255,0), org=(int(_cent[0]), int(_cent[1])))
        
        for fd in facesWithIds:
            _c = fd[1]
            st = (_c[0], _c[1])
            end = (_c[2], _c[3])
            cv2.rectangle(frame, st, end, (255, 0, 0), 2)
        
        self.inventory.update()
        for suspect in self.inventory.suspects:
            #recognition
            if suspect.shouldRecognize():
                recognized = -1
                for embd in suspect.embds:
                    for index, facedata in enumerate(facesWithIds):
                        if fdr.is_match(facedata[3], embd):
                            recognized = index
                            break
                    if recognized >= 0:
                        faceWithId = facesWithIds.pop(recognized)
                        personId = cameraId + "_" + str(faceWithId[2])
                        
                                
                        try:
                            _trackIdIndex = ids.index(faceWithId[2])
                            if not _trackIdIndex > -1:
                                continue
                        except:
                            continue
                            
                        _bbox = bboxes[_trackIdIndex]
                        self.markSuspectOnFrame(frame, suspect, _bbox)
                        
                        suspect.recognized(personId, frame, self.SERVER_ID, cameraId)
                        break
                    
                if recognized == -1:
                    if suspect.last_time_recognized != None:
                        if time.time() - suspect.last_time_recognized > 20:
                            suspect.personId = None
                    
            #displaying the person red on frame
            if suspect.personId != None:
                _trackId = int(suspect.personId.split("_")[1])
                try:
                    _trackIdIndex = ids.index(_trackId)
                    if not _trackIdIndex > -1:
                        continue
                except:
                    continue
                    
                _bbox = bboxes[_trackIdIndex]
                self.markSuspectOnFrame(frame, suspect, _bbox)
        
        
        print("processing one frame: {:2.4f}s".format(t))
        
        
        
        t = time.localtime()
        text = "Server: " + time.strftime("%H:%M:%S", t)
        cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1)
        
        with self.lock:
            camera.processedFrame = frame
            camera.processedFrameTime = time.time()
            self.xo = 1
        return True
        
    def markSuspectOnFrame(self, frame, suspect, _bbox):
        x1,y1,x2,y2 = _bbox[0],_bbox[1],_bbox[2],_bbox[3]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
        text = "Suspect: " + suspect.fullName
        self.textOnFrame(frame, text, (int(x1), int(y1+5)), bc=(0, 0, 255), fc=(0,0,0))
        
    def processFrames(self):
        cameraCurrentlySelected = 0
        keys = list(self.cameras.keys())
        
        self.log("INFO", "Started processing frames", True)
        while self.keepProcessingFrames:
            #stopping thread if no cameras attached
            if len(keys) == 0:
                self.log("WARN", "No cameras attached to this server", True)
                self.stopProcessingFrames()
                continue
                
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
            
        self.log("INFO", "Stopped processing frames", True)

        
    def loadConfig(self):
        self.log("INFO", "Loading config", True)
        confParser = configparser.ConfigParser()
        confParser.read("config.ini")
        
        #get server id
        #information about this server
        self.SERVER_ID = confParser['SERVER']['ID']
        self.SERVER_PORT = confParser['SERVER']['PORT']
        
        
        #web server url
        self.WEB_SERVER = confParser['WEB_SERVER']['URL']
        self.WEB_SERVER_PORT = confParser['WEB_SERVER']['PORT']
        self.log("SUCC", "Done loading", True)
        
    def _loadConfig(self):
        self._removeConfig()
        self._loadConfigFile()
            
    def _removeConfig(self):
        recreate = input("Use existing {} file? (y/n): ".format(Server.CONFIG_FILE_PATH))
        try:
            if recreate == 'n' or recreate == 'N':
                self.log("WARN", "Removing old {}".format(Server.CONFIG_FILE_PATH), True)
                os.remove(Server.CONFIG_FILE_PATH)
        except:
            self.log("EXCEPT", "Error occured while removing", True)
    
    def _loadConfigFile(self):
        if(os.path.exists(Server.CONFIG_FILE_PATH)):
            self.log("INFO", "Loading from {}".format(Server.CONFIG_FILE_PATH), True)
            try:
                config = Jsoner.readJson(Server.CONFIG_FILE_PATH)
                self.WEB_SERVER = config['web_server_url']
                self.SERVER_ID = config['serverId']
            except:
                self.log("EXCEPT", "Couldn't read {}".format(Server.CONFIG_FILE_PATH), True)
                self._createConfigFile()
        else:
            self._createConfigFile()
    
    def _createConfigFile(self):
        self.log("WARN", "Config file not found!", True)
        self.log("INFO", "Creating config", True)
        web_server_url = None
        serverId = None
        while web_server_url == None or len(web_server_url) == 0:
            web_server_url = input("URL of the express-web-server (REST API): ")
        
        while serverId == None or len(serverId) == 0:
            serverId = input("ID of this server assigned by (REST API): ")
        
        conf = {}
        conf['web_server_url'] = web_server_url
        conf['serverId'] = serverId
        
        Jsoner.saveJson(conf, Server.CONFIG_FILE_PATH)
        self._loadConfigFile()
        
        
            
        
        
            
    def loadServerInfoFromWeb(self):
        self.log("INFO", "fetching info from web!", True)
        
        #url = "http://"+self.WEB_SERVER+":"+self.WEB_SERVER_PORT+"/server/"
        _ = '' if self.WEB_SERVER[-1] == '/' else '/'
        url = self.WEB_SERVER + _ +"server/"
        print(url)
        
        #request server details
        try:
            resp = requests.get(url+self.SERVER_ID)
            if resp.status_code != 200:
                self.log("ERR","Could not fetch server details!", True)
                return None
            
            data = resp.json()
            if 'err' in data :
                self.log("ERR", data['err']['message'], True)
                return None
            
            #self.log("DATA", data)
            self.log("SUCC", "info recvd!", True)
            
            self.data = data
            return data
        except:
            self.log("EXCEPT", "Exception while requesting the (REST API) '{}'".format(self.WEB_SERVER), True)
            return None

    def loadSuspects(self, data):
        self.log("INFO", "Loading suspects", True)
        self.suspects = []
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
                    self.log("DOWN", picture_name, True)
                    wget.download(picture_url, out="suspect_pictures/")
                    self.log("SUCC", path + " downloaded", True)
                
                  
                pictures.append(path)
            
            suspect = Suspect(_id, fullName, gender, tags, pictures, self.fdr)
            self.suspects.append(suspect)
            
        self.inventory = Inventory(self.suspects)
        
        self.log("SUCC", "Suspects loaded", True)
        return True
        

    def loadCameras(self, data):
        self.log("INFO", "Loading cameras", True)
        self.cameras = {}
        cameras_data = data['server']['cameras']
        for camera_data in cameras_data:
            _id = camera_data['_id']
            url = camera_data['url']
            lat = camera_data['latitude']
            lng = camera_data['longitude']
            camera = Camera(_id, url, lat, lng)
            self.cameras[_id] = camera
        
            
        self.log("SUCC", "Cameras loaded", True)
        return True
    
    def loadPreprocessingValuesFromWeb(self, data = None):
        if data is None:
            #loads the preprocessing values for the cameras
            self.log("FETCH", "fetching preprocessing values from web", True)
            data = self.loadServerInfoFromWeb()
            self.loadPreprocessingValuesFromWeb(data)
        else:
            self.log("UPDATE", "Updating preprocessing values", True)
            json.dumps(data)
            self.preprocessings = data['server']['preprocessings']
        
    
    '''
    updating
    '''
    def _updateSuspects(self):
        self.log("UPDATE", "Updating suspects", True)
        self.stopProcessingFrames()
        time.sleep(2)
        #updating
        data = self.loadServerInfoFromWeb()
        if data != None:
            if 'err' in data:
                self.log("ERR", data['err']['message'], True)
            else:
                if 'suspects' in data['server']:
                    self.log("INFO", "Saving snapshots of current suspects", True)
                    #current list of suspects being processed
                    _invSuspects = self.inventory.suspects
                    #snapshot of the suspects
                    _invss = {}
                    for _is in _invSuspects:
                        _invss[_is._id] = _is.getSnapShot()
                    
                    self.log("INFO", "Loading new list of suspects", True)
                    self.loadSuspects(data)
                    #new list of suspects being processed
                    self.log("INFO", "Loading snapshots for the already existing suspects", True)
                    _invSuspects = self.inventory.suspects
                    for _is in _invSuspects:
                        if _is._id in _invss:
                            self.log("INFO", "loading snapshot of " + _is.fullName, True)
                            _is.loadSnapShot(_invss[_is._id])
                    self.log("DONE", "Done updating suspects")
        else:
            self.log("ERR", "Empty data from server", True)
        self.startProcessingFrames()
    
    def _updateCameras(self):
        self.log("UPDATE", "Updating cameras", True)
        self.stopProcessingFrames()
        #updating
        data = self.loadServerInfoFromWeb()
        if data != None:
            if 'err' in data:
                self.log("ERR", data['err']['message'], True)
            else:
                if 'cameras' in data['server']:
                    self.log("INFO", "Saving snapshots of current cameras", True)
                    #current list of suspects being processed
                    _cams = self.cameras
                    #snapshot of the suspects
                    _camsss = {}
                    for _id, _cam in _cams.items():
                        _camsss[_id] = _cam.getSnapShot()
                    
                    self.log("INFO", "Loading new list of cameras", True)
                    self.loadCameras(data)
                    #new list of suspects being processed
                    self.log("INFO", "Loading snapshots for the already existing cameras", True)
                    for _id, _cam in self.cameras.items():
                        if _id in _camsss:
                            self.log("INFO", "loading snapshot of " + _cam._id +", url: " + _cam.url , True)
                            if _cam.loadSnapShot(_camsss[_id]):
                                for _s in self.inventory.suspects:
                                    if _s.personId != None:
                                        cId = _s.personId.split("_")[0]
                                        if cId == _cam._id:
                                            self.log("INFO", "Removing tracking from suspect: " + _s.fullName)
                                            _s.personId = None
                    self.log("DONE", "Done updating suspects")
        self.startProcessingFrames()
        
    def _alive(self):
        self.log("IS", "Checking if server has been deleted or not", True)
        self.stopProcessingFrames()
        #updating
        data = self.loadServerInfoFromWeb()
        if data == None or 'err' in data:
            self.log("EXIT", "Stopping the server", True)
            sys.exit()
        else:
            self.startProcessingFrames()
    
    '''
    logger
    '''
    def log(self, _type, msg, printOnTerminal = False):
        Logger._log(_type, msg, printOnTerminal)
        
        
    
            
    '''
    web server
    
    '''
    def startWebServer(self):
        self.log("INFO", "Starting web server", True)
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
            self.log("START", "Starting server", True)
            return make_response("Starting server", 200)
            
            
        @app.route("/stop")
        def stop():
            self.stopProcessingFrames()
            self.log("STOP", "Stopping server", True)
            return make_response("Stopping server", 200)
        
        
        @app.route("/updatep")
        def updatePreprocessingValues():
            self.log("UPDATE", "Updating preprocessing values", True)
            self.loadPreprocessingValuesFromWeb()
            return make_response("Updating preprocessing values", 200)
        
        @app.route("/updatec")
        def updateCameras():
            self.log("UPDATE", "Updating Cameras", True)
            self._updateCameras()
            return make_response("Updating Cameras", 200)
        
        @app.route("/updates")
        def updateSuspects():
            self.log("UPDATE", "Updating Suspects", True)
            self._updateSuspects()
            return make_response("Updating Suspects", 200)
        
        @app.route("/alive")
        def alive():
            self.log("ALIVE", "Checking if server still alive", True)
            self._alive()
            return make_response("Checking if still alive.", 200)
        
        
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
                #print("exception while reading frame")
                frame = np.random.randint(0,256,(h, w,3), dtype=np.uint8)
                
            _, encodedImage = cv2.imencode('.jpg', frame)
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                  bytearray(encodedImage) + 
                  b'\r\n')
        

#fdr = FaceDAndR() 
server = Server()
cv2.destroyAllWindows()

