from ex2_utils import Tracker
import numpy as np
from ex1_utils import gausssmooth
from  ex2_utils import*
class msTracker():
    def initialize(self, image, region):

        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

        self.window = max(region[2], region[3]) * self.parameters.enlarge_factor

        left = max(region[0], 0)
        top = max(region[1], 0)

        right = min(region[0] + region[2], image.shape[1] - 1)
        bottom = min(region[1] + region[3], image.shape[0] - 1)

        self.template = image[int(top):int(bottom), int(left):int(right)]
        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)
        self.size = (region[2], region[3])

    def MeanShiftSeek(self,imag,h,kernelG):
        vel = imag.shape
        patch,mask=get_patch(imag,[vel[0]/2,vel[1]/2],[h,h])
        X=np.arange(-int(h/2),h/2,dtype=int);
        X= np.tile(X,(h,1))
        Y=np.transpose(X)
        for i in range(0,1):
            pTK=patch*kernelG
            Xk=np.sum(X*pTK)/np.sum(pTK);
            Yk = np.sum(Y*pTK)/np.sum(pTK);
            print(Xk)

    def MeanShiftTracker(parameters):
        print("here")
class MSParams():
    def __init__(self):
        self.enlarge_factor = 2

def generate_responses_1():
    responses = np.zeros((100, 100), dtype=np.float32)
    responses[70, 50] = 1
    responses[50, 70] = 0.5
    return gausssmooth(responses, 10)
slika=generate_responses_1()
tracker=msTracker();
tracker.MeanShiftSeek(slika,7,create_epanechnik_kernel(7,7,1));
