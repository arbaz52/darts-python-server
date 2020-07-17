# -*- coding: utf-8 -*-

import cv2
from video_analyser import VideoAnalyser

va = VideoAnalyser()

suspects = ['tmp/arbaz.png','tmp/majid.png']
sEmbds = []

for s in suspects:
    face = va.extractFaces(cv2.imread(s))[0]
    se = va.getEmbedding(face[0])
    sEmbds.append((face[1], se, s))

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    cv2.imshow("Input", frame)
    faces = va.extractFaces(frame)
    
    embeddings = []
    for face in faces:
        embd = va.getEmbedding(face[0])
        embeddings.append((face[0], face[1], embd))
    
    for e in embeddings:
        for se in sEmbds:
            if va.is_match(e[2], se[1]):
                cv2.putText(frame, "Suspect: " + se[2], (e[1][0], e[1][1]), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
                print("Suspect: " + se[2])
    
    
    cv2.imshow("Output", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()