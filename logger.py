import threading
import time

class Logger:    
    @staticmethod
    def _log(_type, msg):
        threading.Thread(target=Logger._logthread, args=(_type, msg)).start()
    
    @staticmethod
    def _logthread(_type, msg):
        string = "[{}]: {} - {}".format(_type, time.strftime("%D %H:%M:%S"), msg)
        #print(string)
        with open("log.txt", "a") as fp:
            fp.write("\n"+string)