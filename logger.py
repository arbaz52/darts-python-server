import threading
import time

class Logger:    
    @staticmethod
    def _log(_type, msg, showOnTerminal=False):
        string = "[{}]: {} - {}".format(_type, time.strftime("%D %H:%M:%S"), msg)
        if showOnTerminal:
            print(string)
        threading.Thread(target=Logger._logthread, args=(string,)).start()
    
    @staticmethod
    def _logthread(s):
        with open("log.txt", "a") as fp:
            fp.write("\n"+s)