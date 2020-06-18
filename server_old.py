import configparser
import requests
from camera import Camera
from suspect import Suspect

import threading
from flask import Flask, Response

from video_analyser import VideoAnalyser

import cv2



app = Flask(__name__)
lock = threading.Lock()


def disconnect(cams):
    #disconnect the links
    for cam in cams:
        cam.cap.release()
    
    cv2.destroyAllWindows()
    



confParser = configparser.ConfigParser()
confParser.read("config.ini")

#get server id
SERVER_ID = confParser['SERVER']['ID']
PORT = confParser['SERVER']['PORT']
#web server url
WEB_SERVER = confParser['WEB_SERVER']['URL']
WEB_SERVER_PORT = confParser['WEB_SERVER']['PORT']

url = "http://"+WEB_SERVER+":"+WEB_SERVER_PORT+"/"

#request server details
resp = requests.get(url+SERVER_ID)
if resp.status_code != 200:
    print("ERR: Could not fetch server details!")
    exit()

data = resp.json()
if 'err' in data :
    print("ERR: " + data['err']['message'])



#initialize video_capture for each camera
cams = []
for camera in data['server']['cameras']:
    print("INFO: Working on camera: " + camera['_id'])
    cam = Camera(camera['_id'], camera['url'], camera['latitude'], camera['longitude'])
    cams.append(cam)
    


#dictionary to access outputframes for each camera
outputFrames = {}
for cam in cams:
    outputFrames[cam._id] = None





#starting video streaming thread
#video_streaming = threading.Thread(target = video_streaming)
#video_streaming.start()

def generate_response(cameraId):
    global outputFrames, lock
    while True:
        #with lock:
        if outputFrames[cameraId] is None:
            print("No frame to output")
            continue
        outputFrame = outputFrames[cameraId]
        (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
        #if cannot generate, skip
        if not flag:
            continue
        
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encodedImage) + 
              b'\r\n')
            
@app.route("/<cameraId>")
def send_frame(cameraId):
    print(cameraId)
    return Response(
            generate_response(cameraId),
            mimetype = "multipart/x-mixed-replace; boundary=frame")

def server():    
    app.run(host="localhost", port=PORT, debug=True,
    		threaded=True, use_reloader=False)
    

serverThread = threading.Thread(target = server)
serverThread.start()
    
            

videoAnalyser = VideoAnalyser()



#calc embeddings for pictures of known suspects
suspects = []
for suspect in data['server']['suspects']:
    sspct = Suspect(suspect['_id'], suspect['fullName'], suspect['gender'], suspect['pictures'])
    sspct.generateEmbeddings(videoAnalyser)
    suspects.append(sspct)
    
exit()

print("INFO: Fetching video streams")
#now for each camera fetch video feed, convert it and make it available for video streaming
while True:
    try:
        for cam in cams:
            if cam.cap.isOpened():
                ret, frame = cam.cap.read()
                
                if not ret:
                    #print("SKIPPING: " + cam._id)
                    continue
            
                #recvd the frame
                print(frame.shape)
                faces = videoAnalyser.extractFaces(frame)
                
                
                
                embeddings = []
                for face in faces:
                    embd = videoAnalyser.getEmbedding(face[0])
                    embeddings.append((face[0], face[1], embd))
                    
                #embeddings of all the faces in frame
                for embd in embeddings:
                    #embeddings of all the suspects in memory
                    for suspect in suspects:
                        #embeddings of all the pictures of suspect
                        for sEmbd in suspect.embeddings:
                            if videoAnalyser.is_match(sEmbd, embd[2]):
                                cv2.putText(frame, "Suspect: " + suspect.fullName, (embd[1][0], embd[1][1]), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
                                
                                print("Suspect: " + suspect.fullName)
                    
                
                outputFrames[cam._id] = frame.copy()
                print("Frame added for: " + cam._id)
                
                '''
                for face in faces:
                    cv2.imshow("Face: ", face[0])
                '''
                
                
                #frame = cv2.resize(frame, (100, 100), cv2.INTER_CUBIC)
                #cv2.imshow(cam._id, frame)
                
                '''
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cam.cap.release()
                    cv2.destroyAllWindows()
                '''
                    
            else:
                print("ERR: Stream not opened for: " + cam._id)

    except Exception as e:
        #or the virtual cameras require restart
        print("ERR: An error occured while streaming")
        print(e)
        disconnect(cams)


#disconnect(cams)