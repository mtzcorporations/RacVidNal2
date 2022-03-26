from ex2_utils import Tracker
import numpy as np
import cv2
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

    def MeanShiftSeek(self,imag,h,kernelG,center,nIter,minChange):
        print(center)
        X=np.arange(-int(h/2),h/2,dtype=int);
        X= np.tile(X,(h,1))
        Y=np.transpose(X)
        p=np.zeros(imag.shape)
        xP=0
        yP=0
        countConver=0
        for i in range(0,1000000):
            patch, mask = get_patch(imag, [center[0] / 2, center[1] / 2], [h, h])
            pK=patch*kernelG
            pKSum=np.sum(pK)
            if (pKSum==0):
                print(i)
                return
            Xk=np.sum(X*pK)/pKSum;
            Yk = np.sum(Y*pK)/pKSum;
            #print(Xk-xP,Yk-yP)
            if(np.abs(Xk-xP) < minChange and np.abs(Yk-yP) < minChange):
                countConver+=1
                if countConver==nIter:
                    print(i)
                    return
            else:
                countConver=0
            center[0] += Xk
            center[1] += Yk
            xP=Xk
            yP=Yk

           # if(Xp <= minChange and Yp<=minChange):
            #    print(i,"Konvergiralo")
            #    break;

        #novo=cv2.bitwise_and(imag,p)
        #cv2.imshow("okno",novo)
        #cv2.waitKey(0)  # wait for a keyboard input
        #cv2.destroyAllWindows()

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
h=11
tracker.MeanShiftSeek(slika,h,create_epanechnik_kernel(h,h,1),[80,70],50,0.00);
