import cv2

class MyVideoCapture:
    
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.idx = -1
        self.end = False
        self.stack = []
        
    def read(self):
        self.idx += 1
        ret, img = self.cap.read()
        if ret:
            self.stack.append(img)
        else:
            self.end = True
        return ret, img
        
    def get_video_clip(self):
        assert len(self.stack) > 0, "clip length must large than 0 !"
        clip = self.stack
        del self.stack
        self.stack = []
        return clip
    
    @property
    def shape(self):
        return int(self.cap.get(3)), int(self.cap.get(4))
    
    def release(self):
        self.cap.release()
