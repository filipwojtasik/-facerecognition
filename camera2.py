from time import time


class Camera(object):
 
    def __init__(self):
        self.frames = [open(f + '.png', 'rb').read() for f in ['face1', 'face2', 'face3','face2', 'face1', 'face4','face5', 'face4', 'face1','face6', 'face1', 'face7','face8','face7']]

    def get_frame(self):
        return self.frames[int(time()) % 14]
