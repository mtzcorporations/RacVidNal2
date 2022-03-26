import numpy as np

from ex1_utils import gausssmooth


def generate_responses_1():
    responses = np.zeros((100, 100), dtype=np.float32)
    responses[70, 50] = 1
    responses[50, 70] = 0.5
    return gausssmooth(responses, 10)
def mST(im,kernelSize):
    for i in range ()

slika=generate_responses_1()

