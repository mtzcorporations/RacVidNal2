from ex2_utils import Tracker
import numpy as np
import cv2
from ex1_utils import gausssmooth
from  ex2_utils import*
class msTracker(Tracker):
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
            print(Xk-xP,Yk-yP)
            if(np.abs(Xk-xP) <= minChange and np.abs(Yk-yP) <= minChange):
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
        #cv2.imshow("okno",novo)
        #cv2.waitKey(0)  # wait for a keyboard input
        #cv2.destroyAllWindows()

    def MeanShiftTracker(delf,image):
        left = max(round(self.position[0] - float(self.window) / 2), 0)
        top = max(round(self.position[1] - float(self.window) / 2), 0)

        right = min(round(self.position[0] + float(self.window) / 2), image.shape[1] - 1)
        bottom = min(round(self.position[1] + float(self.window) / 2), image.shape[0] - 1)

        if right - left < self.template.shape[1] or bottom - top < self.template.shape[0]:
            return [self.position[0] + self.size[0] / 2, self.position[1] + self.size[1] / 2, self.size[0],
                    self.size[1]]

        cut = image[int(top):int(bottom), int(left):int(right)]


class MSParams():
    def __init__(self):
        self.enlarge_factor = 2


slika=generate_responses_1()
parameters = MSParams()
tracker=msTracker(parameters);

h=7
tracker.MeanShiftSeek(slika,h,create_epanechnik_kernel(h,h,1),[80,70],50,0.00);
