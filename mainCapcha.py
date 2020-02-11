#!/usr/bin/env python3
#import socket
#import ssl
import urllib
import urllib.request
from urllib.request import Request, urlopen
import urllib.request
import json
import time
from PIL import Image
import requests
from io import BytesIO

from requests.exceptions import HTTPError
#from imageio import imread
##############################################
from finalOCR import *
################################################

############   SENDING A REQUEST TO THE SERVERS

if __name__ == "__main__":

    # From a round Truth
    avg_list = []
    if len(avg_list) == 0:
        avg_list = roundTruth('CAPCHA')
    # testing for the folder
    #runFolder('CAPCHA',avg_list)
    # text_ = predict_text('CAPCHA/1Z5FIX.png',avg_list)
    # print(text_)

    # file = 'D0BAXU.png'
    # [tempImg_, oriSetImg] = cutImg2(cv2.imread(file,0))
    # print(tempImg_)
    # [tempImg_, oriSetImg] = cutImg(cv2.imread(file,0))
    # print(tempImg_)
    url = 'local.kinggreedy.com:30088/'
    fileName = 'images/1_phplPMeVO.png'
    question = 'api/getCaptchaUnsolved?'
    answer = 'http://local.kinggreedy.com:30088/api/postCaptchaAnswer?' # id=<>&answer=<answer>'

    # http://local.kinggreedy.com:30088/api/getCaptchaUnsolved
    # http://local.kinggreedy.com:30088/images/1_phplPMeVO.png

    print(url+question)
    #response = urllib.request.urlopen(url+question)
    #webContent = response.read()
    #d = json.dumps({"result":[{"id":"1", "url":"tri_deptrai"},{"id":"2", "url":"tri_deptrai2"}]})
    #j = json.loads(d)
    #print(j)
    #list_question = j["result"][0]["id"]
    #print (list_question)
    #list_id = j["result"][0]
    #print (len(j["result"]))
    response = requests.get('http://local.kinggreedy.com:30088/api/getCaptchaUnsolved')
    print (response.content)
    json_request = response.json()
    print(json_request)
    #print (json_request['result'])
    #print(json_request['result']['id'])
    #print(json_request['result']['url'])

    # response_img = requests.get('http://local.kinggreedy.com:30088/'+'images/' + json_request['result']['url'])
    #
    # image = cv2.imread('http://local.kinggreedy.com:30088/'+'images/' + json_request['result']['url'], 0)
    # #img = Image.open(BytesIO(response_img.content))
    # #img = cv2.imread(BytesIO(response_img.content))
    # url_ = 'http://local.kinggreedy.com:30088/'+'images/' + json_request['result']['url']
    #
    # with urllib.request.urlopen('http://local.kinggreedy.com:30088/'+'images/' + json_request['result']['url']) as url:
    #     s = url.read()
    #     # I'm guessing this would output the html source code ?
    #     print(s)
    #     f = open(json_request['result']['url'], "wb")
    #     f.write(s)
    #     f.close()
    #
    #     image = cv2.imread(json_request['result']['url'], 0)
    #     print(image)
    print ("00000000000000000000000000000000000000000000000")
    while 1:
        response = requests.get('http://local.kinggreedy.com:30088/api/getCaptchaUnsolved')
        print(response)
        if response:
            json_request = response.json()
        print (json_request['result'])

        if json_request['result'] == "NULL":
            time.sleep(3)
        else:
            #j = json.loads({"result":{"id":"1", "url":"tri_deptrai"}})
            #list_question = j["result"]["url"]
            #list_id = j["result"]["id"]

            print(type(json_request))
            for index_, v in json_request.items():

                fileName = 'images/' + v["url"]
                #Request(url+fileName, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen('http://local.kinggreedy.com:30088/'+'images/' + v['url']) as url:
                    s = url.read()
                    # I'm guessing this would output the html source code ?
                    print(s)
                    f = open(v['url'], "wb")
                    f.write(s)
                    f.close()
                    text_ = predict_text(v["url"], avg_list)

                # urllib.request.urlretrieve(url+fileName, fileName)
                # text_ = predict_text(index["url"], avg_list)
                print ("send this data: ", v['id'], text_)
                #sendUrl = url+answer
                data = {'id':v['id'], 'answer':text_}
                req = requests.post(url = answer, data = data)

###############################################

"""
HOST, PORT = "bgroup.work", 80
# Create the server, binding to localhost on port 9999
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall('/images/1_phplPMeVO.png')
    data = s.recv(1024)
    print('Received', repr(data))
    s.listen(5)
    conn, addr = s.accept()
    with conn:
        print('Connected by', addr)
        while True:
            data = conn.recv(1024)
            if not data:
                break
            conn.sendall(data)
def cutImage2Boxes(img):
    # convert to gray
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(sum(gray_img))
    # run tesseract, returning the bounding boxes
    boxes = pytesseract.image_to_boxes(gray_img) # also include any config options you use
    h,w = gray_img.shape

    # draw the bounding boxes on the image
    point = 0
    for b in boxes.splitlines():
        b = b.split(' ')
        #tempCrop = gray_img[ h - int(b[2]):h - int(b[4]),(int(b[1])):int(b[3])]
        b = b[1:]
        print (b)
        print (b[0])
        print (b[1])
        print (b[2])
        print (b[3])
        #Croped_image = gray_img[int(b[1]):int(b[1]+b[3]), int(b[0]):int(b[0]+b[2])]

        Croped_image = gray_img[int(b[1]):int(b[1]+b[3]), int(b[0]):int(b[0] + b[2])]
        hc, wc = Croped_image.shape

        charImg = gray_img[int(b[1]):int(b[1]+b[3]), point:w-wc]
        point = w-wc
        #d = gray_img - Croped_image
        #cv2.imshow('', tempCrop)
        #cv2.waitKey(0)

        cv2.imshow('', charImg)
        cv2.waitKey(0)

cutImage2Boxes(img)
"""


# def communicate_():
#     ###/api/postCaptchaAnswer?answer=<answer>
#     ###/api/postCaptchaAnswer?id=<>&answer=<answer>
#     hostname = 'google.com' #local.kinggreedy.com:30088/images'
#     port = 80
#     # PROTOCOL_TLS_CLIENT requires valid cert chain and hostname
#     context = ssl.create_default_context()
#     with socket.create_connection((hostname, port)) as sock:
#         with context.wrap_socket(sock, server_hostname=hostname) as ssock:
#             print(ssock.version())
#

    #if (empty($workingTask)):
    #        return new JsonResponse(["result" => "NULL"]);
    #}

    #s = socket.socket()
    #s.connect((hostname, port))
    #print "Socket successfully created"
    # receive data from the server
    #print s.recv(1024)
    # close the connection
    #s.close()

################################################
# testing
# ima = preprocessingImg(img)
# img = cv2.imread('2YPKA2.png',0)
# img = gaussian_filter(img.astype(float), 4.)
# print (img)
# [tempImg, oriImg] = cutImg(img)
# print (tempImg)
# print (tempImg)
# list_diff = compareCharacter(tempImg, oriImg)

#################avg_list = roundTruth('CAPCHA')
#print (avg_list)
#### testing
#################runFolder('CAPCHA',avg_list)
####
######### wrapper
### https://bgroup.work/api/getCaptchaUnsolved
### https://bgroup.work/images/1_phplPMeVO.png
###/api/postCaptchaAnswer?answer=<answer>
###/api/postCaptchaAnswer?id=<>&answer=<answer>
