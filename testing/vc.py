import cv2
from flask import Flask, Response
from flask import render_template
import numpy as np
import time
app = Flask(__name__)

cap = cv2.VideoCapture(0)
try:
    print("Opening camera")
    while not cap.isOpened():
        print(".", end="")
except:
    print("Camera did not open!")
    exit()
    


(w, h) = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def gen():
    global w, h
    while True:
        frame = None
        try:
            ret, frame = cap.read()
            if not ret:
                print("Empty frame")
                frame = np.random.randint(0,256,(h, w,3), dtype=np.uint8)
        except:
            print("exception while reading frame")
            frame = np.random.randint(0,256,(h, w,3), dtype=np.uint8)
        
        frame = cv2.flip(frame, 1)
        t = time.localtime()
        text = "IPCamera: " + time.strftime("%H:%M:%S", t)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
        
        cv2.imshow("framex", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
        _, encodedImage = cv2.imencode('.jpg', frame)
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encodedImage) + 
              b'\r\n')
        
@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(gen(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")
    
app.run(host='0.0.0.0', port="8082", debug=True,
		threaded=True, use_reloader=False)

cap.release()
cv2.destroyAllWindows()

