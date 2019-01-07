"""
computeFeatures.py

YOUR WORKING FUNCTION for computing features

"""
import numpy as np
import cv2
import matplotlib.pyplot as plt

# you are allowed to import other Python packages above
##########################
def computeFeatures(img):
    # Inputs
    # img: 3-D numpy array of an RGB color image
    #
    # Output
    # featvect: A D-dimensional vector of the input image 'img'
    #
    #########################################################################
    # ADD YOUR CODE BELOW THIS LINE
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.0)
    kps, des = sift.detectAndCompute(gray, None)

    featvect = des
    
    
    # END OF YOUR CODE
    #########################################################################
    return featvect 

# Baseline feature extraction
def computeFeatures_baseline(img):
    # Inputs
    # img: 3-D numpy array of an RGB color image
    #
    # Output
    # featvect: A D-dimensional vector of the input image 'img'
    #
    #########################################################################

    rhist, rbins = np.histogram(img[:,:,0], 64, normed=True)
    ghist, gbins = np.histogram(img[:,:,1], 64, normed=True)
    bhist, bbins = np.histogram(img[:,:,2], 64, normed=True)
    featvect = np.concatenate((rhist, ghist, bhist))

    return featvect 