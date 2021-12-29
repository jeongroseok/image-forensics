"""
Please read the copyright notice located on the readme file (README.md).    
"""
import cv2 as cv
import numpy as np

import functions


def getFingerprint(imgs, sigma=3.):
    L = 4  # number of decomposition levels
    qmf = [.230377813309,	.714846570553, .630880767930, -.027983769417,
           -.187034811719,	.030841381836, .032883011667, -.010597401785]
    qmf /= np.linalg.norm(qmf)

    t = 0
    for img in imgs:
        if t == 0:
            M, N, three = img.shape
            if three == 1:
                continue  # only color images will be processed
            # Initialize sums
            RPsum = np.zeros([M, N, 3], dtype='single')
            # number of additions to each pixel for RPsum
            NN = np.zeros([M, N, 3], dtype='single')

        # The image will be the t-th image used for the reference pattern RP
        t = t+1  # counter of used images

        for j in range(3):
            ImNoise = np.single(functions.NoiseExtract(
                img[:, :, j], qmf, sigma, L))
            Inten = np.multiply(functions.IntenScale(img[:, :, j]),
                                functions.Saturation(img[:, :, j]))    # zeros for saturated pixels
            # weighted average of ImNoise (weighted by Inten)
            RPsum[:, :, j] = RPsum[:, :, j] + np.multiply(ImNoise, Inten)
            NN[:, :, j] = NN[:, :, j] + np.power(Inten, 2)

    del ImNoise, Inten
    RP = np.divide(RPsum, NN + 1)
    RP, LP = functions.ZeroMeanTotal(RP)
    return RP
