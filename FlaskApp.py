import cv2
from flask import  Flask, render_template, Response, jsonify, request, stream_with_context
import time
import threading
from camera2 import Camera as camera2
buf=False

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
    return render_template('index.html')

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
    camera=VideoCamera()
    status = json['status']

    if status == "true":
        app.run()
        return jsonify(result="started")
    else:

        return jsonify(result="stopped")


@app.route('/video_feed')

def video_feed():
    print (buf)
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






if __name__ == '__main__':
    app.run( debug=True)