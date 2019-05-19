import cv2
from flask import  Flask, render_template, Response, jsonify, request, stream_with_context
import time
import threading
from camera2 import Camera as camera2
import numpy as np
import res
import network

buf=False
content="To start click neural network button and wait"
class VideoCamera(object):
    def __init__(self):

        self.video = cv2.VideoCapture(0)
        self.image=0
        self.buf=1
        self.bo=False
        self.i=0
        self.ipocz=1

    def __del__(self):
        self.video.release()

    def get_frame(self):

        success, self.image = self.video.read()
        self.save()


        if self.ipocz <75:
            cv2.imwrite("zdjecia/stream%d.jpg"%(self.i), self.frame2)
        self.i =self.i +1

        if self.i==75:
            self.ipocz=self.ipocz+1
            self.i=self.ipocz
        if self.ipocz==74:
            print ("koniec")

        ret, jpeg = cv2.imencode('.jpg', self.frame2)
        a= jpeg.tobytes()
        return a,self.frame2




    def save(self):

        self.frame2 = cv2.flip(self.image, 1)

        return self.frame2






app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html',content=content)

def gen(camera,buf):

    while True:
        frame, klatka = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def gen2(camera,buf):

    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')



@app.route('/picture', methods=['POST'])
def picture():



    json = request.get_json()

    status = json['status']

    if status == "true":
        app.run()
        return jsonify(result="started")
    else:

        return jsonify(result="stopped")


@app.route('/video_feed')

def video_feed():

    if buf==False:
        return Response(stream_with_context(gen(VideoCamera(),buf)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(gen(VideoCamera(),buf),
                        mimetype='multipart/x-mixed-replace; boundary=frame')




@app.route('/face')

def face():

    return Response((gen2(camera2(), buf)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/net', methods=['POST'])
def net():
    json = request.get_json()

    status = json['status']

    if status == "true":


        model=network.modeling()
        res.resizee()

        numbers = np.random.randint(30, size=(3)) + 10

        img1 = cv2.imread('zdj_resized/%d.jpg' % numbers[0])
        img1 = img1.reshape(1, 200, 200, 3)
        img2 = cv2.imread('zdj_resized/%d.jpg' % numbers[1])
        img2 = img2.reshape(1, 200, 200, 3)
        img3 = cv2.imread('zdj_resized/%d.jpg' % numbers[2])
        img3 = img3.reshape(1, 200, 200, 3)
        img_real = cv2.imread('zdj_resized/74.jpg')
        img_real = img_real.reshape(1, 200, 200, 3)
        inp = [img1, img2]
        inp2 = [img_real, img3]
        pro = model.predict(inp)
        pro2 = model.predict(inp2)


        if (pro2/pro)>0.9 and (pro2/pro)<1.1:
            content = "True"
        else:
            content = "False"

        print(content)

        return jsonify(result="started")
    else:

        return jsonify(result="stopped")



if __name__ == '__main__':
    app.run( debug=True)