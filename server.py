import cv2
from flask import Flask, Response
import numpy as np
import time
from camera import Camera


app = Flask(__name__)

cameras = {"1": ("http://localhost:8082", 31.14, 42.45)}
cams = {i: Camera(i, cameras[i][0], cameras[i][1], cameras[i][2]) for i in cameras.keys()}
outputFrames = {i: None for i in cameras.keys()}


def gen(cameraId):
    if cameraId not in list(cameras.keys()):
        return Response("ERROR")
    #look for camera
    global w, h
    while True:
        try:
            frame = cams[cameraId].get_frame()
            
                
            _, encodedImage = cv2.imencode('.jpg', frame)
        except:
            frame = np.random.randint(0,256,(cams[cameraId].h, cams[cameraId].w,3), dtype=np.uint8)
            _, encodedImage = cv2.imencode('.jpg', frame)
            
        
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encodedImage) + 
              b'\r\n')
        

@app.route("/<cameraId>")
def video_feed(cameraId):
    return Response(gen(cameraId), mimetype='multipart/x-mixed-replace; boundary=frame')

app.run(host='0.0.0.0', port="1234", debug=True,
		threaded=True, use_reloader=False)
