import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

from ex2_utils import Tracker,get_patch,create_epanechnik_kernel,extract_histogram
from ex4_utils import *
import numpy as np
import cv2
import math

#from ex1_utils import gausssmooth
import sympy as sp

class Particle_tracker(Tracker):
    def name(self):
        return 'Particle_filter'

    def initialize(self, image, region):
        self.window = max(region[2], region[3]) * 2
        left = max(region[0], 0)
        top = max(region[1], 0)
        right = min(region[0] + region[2], image.shape[1] - 1)
        bottom = min(region[1] + region[3], image.shape[0] - 1)
        self.template = image[int(top):int(bottom), int(left):int(right)]
        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)
        self.size = (region[2], region[3])

        #NCV za nalogo 2 stepa init
        #Construct a visual model of an object.
        q = 10
        r = 1
        self.Nbins=8
        self.N = 100
        self.sigma=0.6
        self.alpha=0.1

        self.Q, self.A, self.R, self.C = self.NCV_params(q, r)
        #2)

        self.particles = sample_gauss(np.array([self.position[0], self.position[1], 0, 0]), self.Q, self.N)
        self.weights = np.ones(self.particles.shape[0])
s
        #self.weights = np.array([1 / self.N for x in range(self.N)])

        #izbira jedra, zacetnega patcha in histograma
        self.kernel = create_epanechnik_kernel(region[2], region[3], self.sigma)
        self.kShape=self.kernel.shape
        self.patch, inliers = get_patch(image, self.position, self.kShape)
        self.hist = extract_histogram(self.patch, self.Nbins,self.kernel)
        self.hist = np.divide(self.hist, np.sum(self.hist)) #normalize hist

    def track(self,image):
        left = max(round(self.position[0] - float(self.window) / 2), 0)
        top = max(round(self.position[1] - float(self.window) / 2), 0)
        right = min(round(self.position[0] + float(self.window) / 2), image.shape[1] - 1)
        bottom = min(round(self.position[1] + float(self.window) / 2), image.shape[0] - 1)

        if right - left < self.template.shape[1] or bottom - top < self.template.shape[0]:
            return [self.position[0] + self.size[0] / 2, self.position[1] + self.size[1] / 2, self.size[0],
                    self.size[1]]
        noise = sample_gauss(np.zeros(4), self.Q, self.N)
        particleNew_x = []
        particleNew_y = []

        #1 sample new particles
        weights_norm = self.weights / np.sum(self.weights)
        weights_cumsumed = np.cumsum(weights_norm)
        rand_samples = np.random.rand(self.N, 1)
        sampled_idxs = np.digitize(rand_samples, weights_cumsumed)
        self.particles = self.particles[sampled_idxs.flatten(), :]
        #2 move each particle
        for i,p in enumerate(self.particles):
            state = np.matmul(self.A, p)+noise[i]
            particleNew_x.append(state[0])
            particleNew_y.append(state[1])

            patch, mask = get_patch(image, state[0:2], self.kShape)
            hist_new = extract_histogram(patch, self.Nbins, self.kernel)
            hist_new = np.divide(hist_new, np.sum(hist_new)) #no rmalize hist
            hellDist=self.hellingerDist(hist_new,self.hist)
            self.weights[i] = np.exp(-0.5 * hellDist**2 / self.sigma**2)


        self.draw_particles(image, self.particles, self.weights, self.position, self.kShape)

        new_x = np.sum(self.weights/ np.sum(self.weights) * np.array(particleNew_x))
        new_y = np.sum(self.weights/ np.sum(self.weights) * np.array(particleNew_y))

        #Update target visual model
        self.position=(new_x,new_y)
        patch, mask = get_patch(image, self.position, self.kShape)
        new_hist = extract_histogram(patch, self.Nbins, self.kernel)
        new_hist = np.divide(new_hist, np.sum(new_hist)) #normalize hist

        self.hist = (1 - self.alpha) * self.hist + (self.alpha * new_hist)


        left = max(round(self.position[0] - float(self.size[0]) / 2), 0)
        top = max(round(self.position[1] - float(self.size[1]) / 2), 0)


        return [left, top, self.size[0], self.size[1]]  # center x,y,širina,višina

    def NCV_params(self,q_v, r_v):  # constant C=A=1
        T, q = sp.symbols('T q')
        F = sp.Matrix([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
        A = sp.exp(F * T)  # Fi
        L = sp.Matrix([[0, 0], [0, 0], [1, 0], [0, 1]])
        Q = sp.integrate((A * L) * q * (A * L).T, (T, 0, T))

        A = A.subs(T, 1)
        Q = Q.subs(T, 1)
        Q = Q.subs(q, q_v)
        Q = np.array(Q, dtype=float)
        A = np.array(A, dtype=float)

        R = r_v * np.array([[1, 0], [0, 1]])
        C = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

        return Q, A, R, C


    def hellingerDist(self,p, q):
        return math.sqrt(sum([ (math.sqrt(p_i) - math.sqrt(q_i))**2 for p_i, q_i in zip(p, q) ]) / 2)

    def draw_particles(self,image, particles, weights, position, size):
        print(len(particles))
        for (x, y, _, _), weight in zip(particles, weights):
            r = np.random.randint(0, 255)
            g = np.random.randint(0, 255)
            b = np.random.randint(0, 255)
            r=255
            b=0
            g=0
            thickness = 1 #int(weight / 0.002) - 1
            image = cv2.circle(image, (int(x), int(y)), radius=0, color=(b, g, r), thickness=thickness)
        image, _ = get_patch(image, position, (size[0] * 2.5, size[1] * 2.5))
        image = cv2.resize(image, dsize=(int(image.shape[1] * 3), int(image.shape[0] * 3)))
        cv2.imshow("Delci", image)
        cv2.waitKey(0)


class PParams():
    def __init__(self):
        self.enlarge=2



