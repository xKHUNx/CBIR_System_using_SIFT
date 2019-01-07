"""
fullEval.py

DO NOT MODIFY ANY CODES IN THIS FILE
OTHERWISE YOUR RESULTS MAY BE INCORRECTLY EVALUATED! 

@author: John See, 2017
@modified by: Lai Kuan, Wong, 2018

"""
import os
import cv2
import numpy as np
import pickle
import sys, getopt
import matplotlib.pyplot as plt
from computeDistances import computeDistances

# Defaults
dbSize = 500       # number of images in plant database
nPerCat = 50       # number of images in plant database for each category
nC = 10             # number of categories
nRetrieved = 50     # number of images to retrieve
loadFV = True     # flag to indicate if feature vector will be loaded

# Read command line args
myopts, args = getopt.getopt(sys.argv[1:],"r:th")

# parsing command line args
for o, a in myopts:
    if o == '-r':
        nRetrieved = int(a)
        if (nRetrieved > dbSize):
            print("Error: Number of retrieved images exceeds size of database!")
            sys.exit()
    elif o == '-t':          # extract features before evaluating
        cont = input('Caution! Do you wish to continue with feature extraction? (y/n): ')    
        if (cont == 'y'):
            exec(open("featureExtraction.py").read())
            loadFV = False
            print('Done extracting')
        else:
            print("\nCommand aborted. Start over again.")
            sys.exit()
    elif o == '-h':
        print("\nUsage: %s -r numRetrieved    # to specify number of retrieved images" % sys.argv[0])
        print("\n       %s -t         # to enable feature extraction before evaluation" % sys.argv[0])
        print(" ")       
        sys.exit()
    else:
        print(' ')

# if loadFV:
  # load pickled features
base = pickle.load(open("base.pkl", "rb"))
print('Baseline features loaded!')

tfidf = pickle.load(open("tfidf.pkl", "rb"))
print('TF-IDF features loaded!')

bow = pickle.load(open("bow.pkl", "rb"))
print('BoW features loaded!')

print('All features loaded!')
print("")
        
    
# EDIT THIS TO YOUR OWN PATH IF DIFFERENT
# dbpath = 'C:\\Users\\aquas\\Documents\\VIP\\as2\\plantdb'
# dbpath = 'C:\\Users\\Kun Shun\\Documents\\Python\\VIP\\Assignment 2\\plantdb'

# these labels are the class assigned toplan the actual plant names
labels = ('C1','C2','C3','C4','C5','C6','C7','C8','C9','C10')     # BUG FIX 1: changed class label from integer to string

featvect = []  # empty list for holding features
FEtime = np.zeros(dbSize)

# find all pairwise distances
D_base = computeDistances(base)
D_tfidf = computeDistances(tfidf)
D_bow = computeDistances(bow)

# *** Evaluation ----------------------------------------------------------
avg_prec = np.zeros(dbSize)

# =======================================================
# Baseline
# =======================================================

# iterate through all images from each category as query image
for c in range(nC): 
  for i in range(nPerCat):
      idx = (c*nPerCat) + i;

      # access distances of all images from query image, sort them asc
      nearest_idx = np.argsort(D_base[idx, :]);

      # quick way of finding category label for top K retrieved images
      retrievedCats = np.uint8(np.floor((nearest_idx[1:nRetrieved+1])/nPerCat));
 
      # find matches
      hits = (retrievedCats == np.floor(idx/nPerCat))
      
      # calculate average precision of the ranked matches
      if np.sum(hits) != 0:
          avg_prec[idx] = np.sum(hits*np.cumsum(hits)/(np.arange(nRetrieved)+1)) / np.sum(hits)
      else:
          avg_prec[idx] = 0.0
          
mean_avg_prec = np.mean(avg_prec)
mean_avg_prec_perCat = np.mean(avg_prec.reshape(nPerCat, nC), axis=0)
recall = np.sum(hits) / nPerCat

# *** Results & Visualization-----------------------------------------------

print('================================')
print('             Baseline')
print('================================')
print('Mean Average Precision, MAP@%d: %.4f'%(nRetrieved,mean_avg_prec))
print('Recall Rate@%d: \t\t%.4f'%(nRetrieved,recall)) 

plt.figure("Baseline")
x = np.arange(nC)+0.5
plt.xticks(x, list(labels) )
plt.xlim([0,10]), plt.ylim([0,1])
markerline, stemlines, baseline = plt.stem(x, mean_avg_prec_perCat, '-.')
plt.grid(True)
plt.xlabel('Plant species'), plt.ylabel('MAP per species')
print("")


# =======================================================
# BoW
# =======================================================

# iterate through all images from each category as query image
for c in range(nC): 
  for i in range(nPerCat):
      idx = (c*nPerCat) + i;

      # access distances of all images from query image, sort them asc
      nearest_idx = np.argsort(D_bow[idx, :]);

      # quick way of finding category label for top K retrieved images
      retrievedCats = np.uint8(np.floor((nearest_idx[1:nRetrieved+1])/nPerCat));
 
      # find matches
      hits = (retrievedCats == np.floor(idx/nPerCat))
      
      # calculate average precision of the ranked matches
      if np.sum(hits) != 0:
          avg_prec[idx] = np.sum(hits*np.cumsum(hits)/(np.arange(nRetrieved)+1)) / np.sum(hits)
      else:
          avg_prec[idx] = 0.0
          
mean_avg_prec = np.mean(avg_prec)
mean_avg_prec_perCat = np.mean(avg_prec.reshape(nPerCat, nC), axis=0)
recall = np.sum(hits) / nPerCat

# *** Results & Visualization-----------------------------------------------

print('================================')
print('           Bag-of-word')
print('================================')
print('Mean Average Precision, MAP@%d: %.4f'%(nRetrieved,mean_avg_prec))
print('Recall Rate@%d: \t\t%.4f'%(nRetrieved,recall)) 

plt.figure("Bag-of-word")
x = np.arange(nC)+0.5
plt.xticks(x, list(labels) )
plt.xlim([0,10]), plt.ylim([0,1])
markerline, stemlines, baseline = plt.stem(x, mean_avg_prec_perCat, '-.')
plt.grid(True)
plt.xlabel('Plant species'), plt.ylabel('MAP per species')

# =======================================================
# TF-IDF
# =======================================================

# iterate through all images from each category as query image
for c in range(nC): 
  for i in range(nPerCat):
      idx = (c*nPerCat) + i;

      # access distances of all images from query image, sort them asc
      nearest_idx = np.argsort(D_tfidf[idx, :]);

      # quick way of finding category label for top K retrieved images
      retrievedCats = np.uint8(np.floor((nearest_idx[1:nRetrieved+1])/nPerCat));
 
      # find matches
      hits = (retrievedCats == np.floor(idx/nPerCat))
      
      # calculate average precision of the ranked matches
      if np.sum(hits) != 0:
          avg_prec[idx] = np.sum(hits*np.cumsum(hits)/(np.arange(nRetrieved)+1)) / np.sum(hits)
      else:
          avg_prec[idx] = 0.0
          
mean_avg_prec = np.mean(avg_prec)
mean_avg_prec_perCat = np.mean(avg_prec.reshape(nPerCat, nC), axis=0)
recall = np.sum(hits) / nPerCat
print("")

# *** Results & Visualization-----------------------------------------------
print('================================')
print('              TF-IDF')
print('================================')
print('Mean Average Precision, MAP@%d: %.4f'%(nRetrieved,mean_avg_prec))
print('Recall Rate@%d: \t\t%.4f'%(nRetrieved,recall)) 

plt.figure("TF-IDF")
x = np.arange(nC)+0.5
plt.xticks(x, list(labels) )
plt.xlim([0,10]), plt.ylim([0,1])
markerline, stemlines, baseline = plt.stem(x, mean_avg_prec_perCat, '-.')
plt.grid(True)
plt.xlabel('Plant species'), plt.ylabel('MAP per species')


#fig, axs = plt.subplots(2, 5, figsize=(12, 6), facecolor='w', edgecolor='w')
#fig.subplots_adjust(hspace = .5, wspace=.001)
#axs = axs.ravel()
#for i in range(nC):
#    imgfile = os.path.join(dbpath, str(nearest_idx[i+1]) + '.jpg')
#    matched_img = cv2.cvtColor(cv2.imread(imgfile), cv2.COLOR_BGR2RGB)
#    axs[i].imshow(matched_img)
#    axs[i].set_title(str(i+1) + '. ' + labels[retrievedCats[i]])
#    axs[i].set_xticks([])
#    axs[i].set_yticks([])

plt.show()
      