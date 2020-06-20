import requests

url = "http://localhost:3000/server/1/alert/"
files ={'frame':open('./tmp/majid.png','rb')}
d = {"test":"data"}
r = requests.post(url,files=files, data=d)
print(r.json())