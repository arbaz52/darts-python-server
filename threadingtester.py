import threading
import time
class Ser:
    def __init__(self):
        self.x = 0
        self.lock = threading.Lock()
    
    def start_threads(self):
        threading.Thread(target = self.updater, args=()).start()
        threading.Thread(target = self.reader, args=()).start()
    
    def reader(self):
        for i in range(10):
            print("reader waiting for lock")
            with self.lock:
                print("reader showing: ")
                print(self.x)
            print("reading releasing lock")
            
    def updater(self):
        for i in range(10):
            print("updater waiting for lock")
            with self.lock:
                self.x += 1
                print("updater updating: ")
                time.sleep(1)
            print("updater releasing lock")
        

s = Ser()
s.start_threads()