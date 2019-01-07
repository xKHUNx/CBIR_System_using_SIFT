"""
featureExtraction.py

DO NOT MODIFY ANY CODES IN THIS FILE
OTHERWISE YOUR RESULTS MAY BE INCORRECTLY EVALUATED! 
I MODIFIED IT WITH PERMISSION FROM WONG. XD

@author: John See, 2017
@modified by: Lai Kuan, Wong, 2018
@modified by: Kun Shun, Goh, 2018

"""
import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time
from scipy.cluster.vq import kmeans, vq
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer
from computeFeatures import computeFeatures, computeFeatures_baseline

# EDIT THIS TO YOUR OWN PATH IF DIFFERENT
# dbpath = 'C:\\Users\\aquas\\Documents\\VIP\\as2\\plantdb'
dbpath = 'C:\\Users\\Kun Shun\\Documents\\Python\\VIP\\Assignment 2\\plantdb\\train'

##############################################################################

# List of features that stores
feat = []
base_feat = []

for idx in range(500):
    # Load and convert image
    img = cv2.imread( os.path.join(dbpath, str(idx+1) + ".jpg") )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Compute SIFT features for each keypoints
    feat.append(computeFeatures(img))

    # Compute baseline features for each image
    base_feat.append(computeFeatures_baseline(img))

    print('Extracting features for image #%d'%idx )

# Stack all features together
alldes = np.vstack(feat)

k = 50
"""
# Perform K-means clustering
alldes = np.float32(alldes)      # convert to float, required by kmeans and vq functions
e0 = time.time()
codebook, distortion = kmeans(alldes, k)
code, distortion = vq(alldes, codebook)
e1 = time.time()
print("Time to build {}-cluster codebook from {} images: {} seconds".format(k,alldes.shape[0],e1-e0))

# Save codebook as pickle file
pickle.dump(codebook, open("codebook.pkl", "wb"))

"""

# Load cookbook
codebook = pickle.load(open("codebook.pkl", "rb"))

##############################################################################

# these labels are the classes assigned to the actual plant names
labels = ('C1','C2','C3','C4','C5','C6','C7','C8','C9','C10')     # BUG FIX 1: changed class label from integer to string


#====================================================================
# Bag-of-word Features
#====================================================================
# Create Bag-of-word list
bow = []

# Get label for each image, and put into a histogram (BoW)
for f in feat:
    code, distortion = vq(f, codebook)
    bow_hist, _ = np.histogram(code, k, normed=True)
    bow.append(bow_hist)
    
# Stack them together
temparr = np.vstack(bow)

# Put them into feature vector
fv = np.reshape(temparr, (temparr.shape[0], temparr.shape[1]) )
del temparr


# pickle your features (bow)
pickle.dump(fv, open("bow.pkl", "wb"))
print('')
print('Bag-of-words features pickled!')

#====================================================================
# TF-IDF Features
#====================================================================
def tfidf(bow):
	# td-idf weighting
    transformer = TfidfTransformer(smooth_idf=True)
    t = transformer.fit_transform(bow).toarray()
        
    # normalize by Euclidean (L2) norm before returning 
    t = normalize(t, norm='l2', axis=1)
    
    return t

# re-run vq without normalization, normalize after computing tf-idf
bow = np.vstack(bow)
t = tfidf(bow)

# pickle your features (tfidf)
pickle.dump(t, open("tfidf.pkl", "wb"))
print('TF-IDF features pickled!')

#====================================================================
# Baseline Features
#====================================================================
# Stack all features together
base_feat = np.vstack(base_feat)

# pickle your features (baseline)
pickle.dump(base_feat, open("base.pkl", "wb"))
print('Baseline features pickled!')

#====================================================================
