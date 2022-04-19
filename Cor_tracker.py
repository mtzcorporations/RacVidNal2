import matplotlib.pyplot as plt

from ex2_utils import Tracker
import numpy as np
import cv2
import math
#from ex1_utils import gausssmooth
from  ex2_utils import get_patch

class MOOSE_tracker(Tracker):
    def name(self):
        return 'MOOSE_fake'

    def initialize(self, image, region):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.alpha = 0.1  # [0.7,0.9,]
        self.sigma = 2
        self.englare = 2
        self.window = max(region[2], region[3]) * 2
        left = max(region[0], 0)
        top = max(region[1], 0)
        right = min(region[0] + region[2], image.shape[1] - 1)
        bottom = min(region[1] + region[3], image.shape[0] - 1)
        self.template = image[int(top):int(bottom), int(left):int(right)]
        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)
        self.size = (region[2], region[3])

        #newSize = (self.size[0] * self.englare, self.size[1] * self.englare)
        xS=(self.size[0]*self.englare,self.size[1] * self.englare)
        self.G = self.create_gauss_peak(xS, self.sigma)
        #plt.imshow(self.G)
        #plt.show()
        self.sz=(self.G.shape[1],self.G.shape[0])
        patch,mask = get_patch(image, self.position, self.sz)  # returns patch,mask
        #plt.imshow(patch)
        #plt.show()
        patch = patch * self.create_cosine_window(patch.shape)
        self.F = patch
        self.H_t = self.H_filter(self.F) #size[0],size[1]
        self.params=MooseParams()

    def track(self,image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print(self.params.enlarge_factor)
        left = max(round(self.position[0] - float(self.window) / 2), 0)
        top = max(round(self.position[1] - float(self.window) / 2), 0)
        right = min(round(self.position[0] + float(self.window) / 2), image.shape[1] - 1)
        bottom = min(round(self.position[1] + float(self.window) / 2), image.shape[0] - 1)

        if right - left < self.template.shape[1] or bottom - top < self.template.shape[0]:
            return [self.position[0] + self.size[0] / 2, self.position[1] + self.size[1] / 2, self.size[0],
                    self.size[1]]

        patch,mask = get_patch(image, self.position, self.sz)
        patch = patch * self.create_cosine_window(patch.shape)
        self.F = patch
        R =np.real(self.R_localization_step(self.H_t, self.F))
        H = self.H_filter(self.F)
        self.H_t = self.H_t_update(self.H_t, H, 0.1)

        i2 = np.unravel_index(np.argmax(R, axis=None), R.shape)
        x = i2[1]
        y = i2[0]
        if x > (self.size[0] *self.englare) / 2:
             x = x - self.size[0] * self.englare
        if y > (self.size[1] * self.englare) / 2:
             y = y - self.size[1] * self.englare
        # if x > (self.sz[1]) / 2:
        #     x = x - self.sz[1]
        # if y > (self.sz[0]) / 2:
        #     y = y - self.sz[0]

        x = x + self.position[0]
        y = y + self.position[1]
        self.position = (x, y)

        left = max(round(self.position[0] - float(self.size[0]) / 2), 0)
        top = max(round(self.position[1] - float(self.size[1]) / 2), 0)

        return [left, top, self.size[0], self.size[1]]  # center x,y,širina,višina


    def H_filter(self,F):
        lb = 0.0001
        F = np.fft.fft2(F)
        Fc = np.conjugate(F)

        G_F = np.fft.fft2(self.G)
        H_c = (G_F * Fc) / (F * Fc + lb)
        return H_c

    def R_localization_step(self,H, F):
        F = np.fft.fft2(F)
        R = np.fft.ifft2(H * F)
        return R

    def H_t_update(self,Ht_1, H_temp, alpha):
        return (1 - alpha) * Ht_1 + alpha * H_temp

    def create_gauss_peak(self,target_size, sigma):
        # target size is in the format: (width, height)
        # sigma: parameter (float) of the Gaussian function
        # note that sigma should be small so that the function is in a shape of a peak
        # values that make sens are approximately from the interval: ~(0.5, 5)
        # output is a matrix of dimensions: (width, height)
        w2 = math.floor(target_size[0] / 2)
        h2 = math.floor(target_size[1] / 2)
        [X, Y] = np.meshgrid(np.arange(-w2, w2 + 1), np.arange(-h2, h2 + 1))
        G = np.exp(-X ** 2 / (2 * sigma ** 2) - Y ** 2 / (2 * sigma ** 2))
        G = np.roll(G, (-h2, -w2), (0, 1))
        return G

    def create_cosine_window(self,target_size):
        # target size is in the format: (width, height)
        #print(type(target_size[0]))
        # output is a matrix of dimensions: (width, height)
        return cv2.createHanningWindow(((target_size[1]), (target_size[0])), cv2.CV_32F)


class MooseParams():
    def __init__(self,a,s,e):
        self.alpha = a
        self.sigma = s
        self.enlarge_factor = e


