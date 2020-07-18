import os
import json

class Jsoner:
    @staticmethod
    def readJson(filepath):
        exception = None
        if not os.path.exists(filepath):
            exception = Exception("File does not exist!")
            raise exception
        try:
            fp = open(filepath, 'r')
            try:
                content = json.load(fp)
            except:
                fp.close()
                exception = Exception("Invalid format in file")
                raise exception
            fp.close()
            return content
        except:
            if exception is not None:
                raise exception
            raise Exception("Invalid file")
            
    @staticmethod
    def saveJson(jsondict, filepath):
        exception = None
        try:
            fp = open(filepath, 'w')
            
            try:
                json.dump(jsondict, fp)
            except:
                fp.close()
                exception = Exception("couldn't store to file")
                raise exception
            fp.close()
        except:
            if exception is not None:
                raise exception
            raise Exception("Couldn't store the json")

        return True
    