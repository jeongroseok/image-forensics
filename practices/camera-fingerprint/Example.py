import src.Functions as Fu
import src.Filter as Ft
import src.getFingerprint as gF
import src.maindir as md
import src.extraUtils as eu
import numpy as np
import os
import cv2 as cv


# extracting Fingerprint from same size images in a path
Images = [fr'train\iPhone-4s\(iP4s){i+1}.jpg' for i in range(5)]

RP,_,_ = gF.getFingerprint(Images)
RP = Fu.rgb2gray1(RP)
sigmaRP = np.std(RP)
Fingerprint = Fu.WienerInDFT(RP, sigmaRP)

# To save RP in a '.mat' file:
#import scipy.io as sio
#sio.savemat('Fingerprint.mat', {'RP': RP, 'sigmaRP': sigmaRP, 'Fingerprint': Fingerprint})

def f(imx):
    Noisex = Ft.NoiseExtractFromImage(imx, sigma=2.)
    Noisex = Fu.WienerInDFT(Noisex, np.std(Noisex))

    # The optimal detector (see publication "Large Scale Test of Sensor Fingerprint Camera Identification")
    Ix = cv.cvtColor(cv.imread(imx),# image in BGR format
                    cv.COLOR_BGR2GRAY)

    C = Fu.crosscorr(Noisex,np.multiply(Ix, Fingerprint))
    det, det0 = md.PCE(C)
    for key in det.keys(): print("{0}: {1}".format(key, det[key]))
    eu.mesh(C)
#f()
#f(im1)
#f(im2)
f(fr'train\iPhone-4s\(iP4s)10.jpg')
f(fr'train\iPhone-4s\(iP4s)30.jpg')
f(fr'train\iPhone-4s\(iP4s)100.jpg')
f(fr'train\iPhone-6\(iP6)1.jpg')
print('done')

