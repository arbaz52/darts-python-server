import argparse
'''
parser = argparse.ArgumentParser()
parser.add_argument("filepath")

args = parser.parse_args()
filepath = args.filepath
'''
filepath = "files/suspects.mp4"

import threading
lock = threading.Lock()
import cv2
from flask import Flask, Response
import numpy as np
import time
app = Flask(__name__)

cap = cv2.VideoCapture(filepath)
try:
    print("Opening camera -> " + filepath)
except:
    print("Camera did not open!")
    exit()
    
outputFrame = None
(w, h) = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
s = 0.3

def read_frame_thread():
    global outputFrame, lock, filepath
    cap = cv2.VideoCapture(filepath)
    while True:
        time.sleep(0.1)
        if not cap.isOpened():
            cap = cv2.VideoCapture(filepath)
        frame = None
        try:
            ret, frame = cap.read()
            if not ret:
                print("Empty frame")
                frame = np.random.randint(0,256,(h, w,3), dtype=np.uint8)
                cap = cv2.VideoCapture(filepath)
        except:
            print("exception while reading frame")
            frame = np.random.randint(0,256,(h, w,3), dtype=np.uint8)
            
        frame = cv2.resize(frame, (int(w*s), int(h*s)), cv2.INTER_CUBIC)
        frame = cv2.flip(frame, 1)
        with lock:
            outputFrame = frame
        

def gen():
    global w, h, s, cap, lock
    while True:
        frame = None
        with lock:
            frame = outputFrame
        if frame is not None:
            t = time.localtime()
            text = "IPCamera: " + time.strftime("%H:%M:%S", t)
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
            _, encodedImage = cv2.imencode('.jpg', frame)
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                  bytearray(encodedImage) + 
                  b'\r\n')
        

@app.route("/")
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


threading.Thread(target=read_frame_thread).start()

app.run(host='0.0.0.0', port="8082", debug=True,
		threaded=True, use_reloader=False)

cap.release()
cv2.destroyAllWindows()

