import cv2
import numpy as np



class Preprocessing:
    @staticmethod
    def putText(frame, text, org):
        cv2.putText(frame, text=text, fontFace=cv2.FONT_HERSHEY_COMPLEX, org=org, color=(0, 0, 255), fontScale=0.5)
    
    #frame: input frame
    #alpha: weight for original image
    #beta: weight for blurred image (calculated)
    @staticmethod
    def sharpenImage(frame, alpha = 2, beta = -1, k = None):
        if k is not None:
            alpha = k + 1
            beta = -k
        gauss = cv2.GaussianBlur(frame, (7, 7), 0)
        unsharp_image = cv2.addWeighted(frame, alpha, gauss, beta, 0)
        return unsharp_image
    
    @staticmethod
    def denoiseImage(frame, strength=3, templateWindowSize = 7, searchWindowSize = 21):
        return cv2.fastNlMeansDenoisingColored(frame,
                                               None,
                                               strength,
                                               strength,
                                               templateWindowSize,
                                               searchWindowSize)
    
    @staticmethod
    def adjustBrightness(frame, brightness=1):
        return cv2.addWeighted(frame, brightness, frame, brightness, 0)
    
'''
cap = cv2.VideoCapture(0)
alpha = 2
beta = -1
k = 1
b = 1
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    
    
    i = cv2.waitKey(1) & 0xFF
    if i == ord('q'):
        break
    
    if i == ord('w'):
        alpha += 0.1
    if i == ord('s'):
        alpha -= 0.1
        
    if i == ord('e'):
        beta += 0.1
    if i == ord('d'):
        beta -= 0.1
    if i == ord('r'):
        k += 0.1
    if i == ord('f'):
        k -= 0.1
        
        
    if i == ord('t'):
        b += 0.1
    if i == ord('g'):
        b -= 0.1
    
        
    
    frame = cv2.flip(frame, 1)
    #shape = (300, 200)
    #frame = cv2.resize(frame, shape)
    
    cv2.imshow("frame", frame)
    
    sharp_image = Preprocessing.sharpenImage(frame, k=k)
    text = "alpha: {}, beta: {}, k: {}, brightness: {}".format(alpha, beta, k, b)
    cv2.putText(sharp_image, text=text, fontFace=cv2.FONT_HERSHEY_COMPLEX, org=(5, 30), color=(0, 0, 255), fontScale=0.5)
    cv2.imshow("sharpened", sharp_image)
    noise = np.random.random(frame.shape) * 255
    noise = noise.astype("uint8")
    cv2.imshow("noise", noise)
    
    noisey_frame = cv2.addWeighted(frame, 0.5, noise, 0.5, 0)
    cv2.imshow("noisey frame", noisey_frame)
    
    dst = Preprocessing.denoiseImage(frame)
    cv2.imshow("Denoised image", dst)
    
    sharp_image = Preprocessing.sharpenImage(dst, k=k)
    cv2.imshow("Denoised sharp", sharp_image)
    bright_image = Preprocessing.adjustBrightness(sharp_image, b)
    cv2.imshow("brightness", bright_image)
    
cap.release()
cv2.destroyAllWindows()

'''