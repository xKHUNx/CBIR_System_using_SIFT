"""
queryEval.py

DO NOT MODIFY ANY CODES IN THIS FILE
OTHERWISE YOUR RESULTS MAY BE INCORRECTLY EVALUATED! 
I DON'T CARE YOUR CODE SO BUGGY


@author: John See, 2017
@modified by: Lai Kuan, Wong, 2018
@fixed by: Kun Shun, Goh, 2018

"""
import os
import cv2
import numpy as np
import pickle
import sys, getopt
import matplotlib.pyplot as plt
from scipy.cluster.vq import vq
from computeFeatures import computeFeatures, computeFeatures_baseline
from computeDistances import computeDistances

# EDIT THIS TO YOUR OWN PATH IF DIFFERENT
# dbpath = 'C:\\Users\\aquas\\Documents\\VIP\\as2\\plantdb'
dbpath = 'C:\\Users\\Kun Shun\\Documents\\Python\\VIP\\Assignment 2\\plantdb\\train'
# dbpath= 'C:\\Users\\Koh\\Desktop\\1151101808_Assignment-2\\plantdb\\train'

# these labels are the classes assigned to the actual plant names
labels = ('C1','C2','C3','C4','C5','C6','C7','C8','C9','C10')     # BUG FIX 1: changed class label from integer to string

# create a arrays for precision and recall
precision_bow = []
precision_tfidf = []
precision_base = []
recall_bow = []
recall_tfidf = []
recall_base = []

# Read command line args
myopts, args = getopt.getopt(sys.argv[1:],"d:q:h")

# parsing command line args
for o, a in myopts:
    if o == '-d':
        queryfile = os.path.join(dbpath, a + '.jpg')
        gt_idx = np.uint8(np.floor((int(a)-1)/50))
        if not os.path.isfile(queryfile):
            print("Error: Query file does not exist! Please check.")
            sys.exit()
    elif o == '-q':
        queryfile = a
        if not os.path.isfile(queryfile):
            print("Error: Query file does not exist! Please check.")
            sys.exit()
        # tokenize filename to get category label and index
        gt = str(queryfile.split("_")[1]).split(".")[0]
        gt_idx = labels.index(gt)
    elif o == '-h':
        print("\nUsage: %s -d dbfilenumber\n       # to specify a single query image from the database for evaluation" % sys.argv[0])
        print("\n       %s -q queryfile\n       # to specify a new query image for evaluation" % sys.argv[0])
        print(" ")       
        sys.exit()
    else:
        print(' ')
    

# read query image file
img = cv2.imread(queryfile)
query_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


plt.figure("Query Image")
# show stuff
plt.imshow(query_img), plt.title('Query image: %s'%labels[gt_idx])
plt.xticks([]), plt.yticks([])
print('Query image: %s'%labels[gt_idx])
print("")

#====================================================================
# Bag-of-word Features
#====================================================================

featvect = []  # empty list for holding features
FEtime = np.zeros(500)

# load pickled features
fv = pickle.load(open("bow.pkl", "rb") )
print('BoW features loaded')

# Compute features
newfeat = computeFeatures(query_img)
# Load cookbook
codebook = pickle.load(open("codebook.pkl", "rb"))
code, distortion = vq(newfeat, codebook)
# Map features to label and obtain BoW
k = codebook.shape[0]
bow_hist, _ = np.histogram(code, k, normed=True)
# Update newfeat to BoW
newfeat = bow_hist


# insert new feat to the top of the feature vector stack
fv = np.insert(fv, 0, newfeat, axis=0)

# find all pairwise distances
D = computeDistances(fv)


# *** Evaluation ----------------------------------------------------------

# number of images to retrieve
nRetrieved = 50

# access distances of all images from query image (first image), sort them asc
nearest_idx = np.argsort(D[0, :]);

# quick way of finding category label for top K retrieved images
retrievedCats = np.uint8(np.floor(((nearest_idx[1:nRetrieved+1])-1)/50));

#for curve  500
retrievedCurve = np.uint8(np.floor(((nearest_idx[1:500+1])-1)/50));
 
# find matches
hits_q = (retrievedCats == gt_idx)
  
# calculate average precision of the ranked matches
if np.sum(hits_q) != 0:
  avg_prec_q = np.sum(hits_q*np.cumsum(hits_q)/(np.arange(nRetrieved)+1)) / np.sum(hits_q)
else:
  avg_prec_q = 0.0
          
recall = np.sum(hits_q) / 50        #nRetrieved

# *** Results & Visualization-----------------------------------------------

print('================================')
print('           Bag-of-word')
print('================================')
print('Average Precision, AP@%d: %.4f'%(nRetrieved,avg_prec_q))
print('Recall Rate@%d: \t  %.4f'%(nRetrieved,recall)) 


fig, axs = plt.subplots(2, 5, figsize=(15, 6), facecolor='w', edgecolor='w', num='Bag-of-word')
fig.subplots_adjust(hspace = .5, wspace=.001)
fig.suptitle('Bag-of-word', fontsize=16)
axs = axs.ravel()
#to calculate how many match image for precision = match / return
match_precision=0
for i in range(10):
    imgfile = os.path.join(dbpath, str(nearest_idx[i+1]) + '.jpg')
    matched_img = cv2.cvtColor(cv2.imread(imgfile), cv2.COLOR_BGR2RGB)
    axs[i].imshow(matched_img)
    axs[i].set_title(str(str(i+1) + '. ' + str(labels[retrievedCats[i]])))
    axs[i].set_xticks([])
    axs[i].set_yticks([])
  
for i in range(500):
	if(labels[gt_idx] == labels[retrievedCurve[i]]):
		match_precision = match_precision + 1
	temp_return = i + 1
	temp = match_precision / temp_return
	precision_bow.append(temp)
	temp_recall = match_precision/50   #50 is number of relevant image # each query has 50 images
	recall_bow.append(temp_recall)

#print(precision_bow)
#print("")
#print(recall_bow)
print("")


#====================================================================
# TD-IDF Features
#====================================================================

featvect = []  # empty list for holding features
FEtime = np.zeros(500)

# load pickled features
fv = pickle.load(open("tfidf.pkl", "rb") )
print('TF-IDF features loaded')

# read query image file
img = cv2.imread(queryfile)
query_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# Compute features
newfeat = computeFeatures(query_img)
# Load cookbook
codebook = pickle.load(open("codebook.pkl", "rb"))
code, distortion = vq(newfeat, codebook)
# Map features to label and obtain BoW
k = codebook.shape[0]
bow_hist, _ = np.histogram(code, k, normed=True)
# Update newfeat to BoW
newfeat = bow_hist


# insert new feat to the top of the feature vector stack
fv = np.insert(fv, 0, newfeat, axis=0)

# find all pairwise distances
D = computeDistances(fv)


# *** Evaluation ----------------------------------------------------------

# number of images to retrieve
nRetrieved = 50

# access distances of all images from query image (first image), sort them asc
nearest_idx = np.argsort(D[0, :]);

# quick way of finding category label for top K retrieved images
retrievedCats = np.uint8(np.floor(((nearest_idx[1:nRetrieved+1])-1)/50));

#for curve  500
retrievedCurve = np.uint8(np.floor(((nearest_idx[1:500+1])-1)/50));
 
# find matches
hits_q = (retrievedCats == gt_idx)
  
# calculate average precision of the ranked matches
if np.sum(hits_q) != 0:
  avg_prec_q = np.sum(hits_q*np.cumsum(hits_q)/(np.arange(nRetrieved)+1)) / np.sum(hits_q)
else:
  avg_prec_q = 0.0
          
recall = np.sum(hits_q) / 50 #NumberofRetrieved

# *** Results & Visualization-----------------------------------------------

print('================================')
print('              TF-IDF')
print('================================')
print('Average Precision, AP@%d: %.4f'%(nRetrieved,avg_prec_q))
print('Recall Rate@%d: \t  %.4f'%(nRetrieved,recall)) 


fig, axs = plt.subplots(2, 5, figsize=(15, 6), facecolor='w', edgecolor='w', num='TF-IDF')
fig.subplots_adjust(hspace = .5, wspace=.001)
fig.suptitle('TF-IDF', fontsize=16)
axs = axs.ravel()
#to calculate how many match image for precision = match / return
match_precision=0
for i in range(10):
    imgfile = os.path.join(dbpath, str(nearest_idx[i+1]) + '.jpg')
    matched_img = cv2.cvtColor(cv2.imread(imgfile), cv2.COLOR_BGR2RGB)
    axs[i].imshow(matched_img)
    axs[i].set_title(str(str(i+1) + '. ' + str(labels[retrievedCats[i]])))
    axs[i].set_xticks([])
    axs[i].set_yticks([])
  
for i in range(500):
	if(labels[gt_idx] == labels[retrievedCurve[i]]):
		match_precision = match_precision + 1
	temp_return = i + 1
	temp = match_precision / temp_return
	precision_tfidf.append(temp)
	temp_recall = match_precision/50   #50 is number of relevant image # each query has 50 images
	recall_tfidf.append(temp_recall)

#print(precision_tfidf)
#print("")
#print(recall_tfidf)
print("")

#====================================================================
# Baseline Features
#====================================================================

featvect = []  # empty list for holding features
FEtime = np.zeros(500)

# load pickled features
fv = pickle.load(open("base.pkl", "rb") )
print('Baseline features loaded')

# read query image file
img = cv2.imread(queryfile)
query_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# Compute features
newfeat = computeFeatures_baseline(query_img)

# insert new feat to the top of the feature vector stack
fv = np.insert(fv, 0, newfeat, axis=0)

# find all pairwise distances
D = computeDistances(fv)


# *** Evaluation ----------------------------------------------------------

# number of images to retrieve
nRetrieved = 50

# access distances of all images from query image (first image), sort them asc
nearest_idx = np.argsort(D[0, :]);

# quick way of finding category label for top K retrieved images
retrievedCats = np.uint8(np.floor(((nearest_idx[1:nRetrieved+1])-1)/50));

#for curve  500
retrievedCurve = np.uint8(np.floor(((nearest_idx[1:500+1])-1)/50));
 
# find matches
hits_q = (retrievedCats == gt_idx)
  
# calculate average precision of the ranked matches
if np.sum(hits_q) != 0:
  avg_prec_q = np.sum(hits_q*np.cumsum(hits_q)/(np.arange(nRetrieved)+1)) / np.sum(hits_q)
else:
  avg_prec_q = 0.0
          
recall = np.sum(hits_q) / 50  #nRetrieved

# *** Results & Visualization-----------------------------------------------

print('================================')
print('             Baseline')
print('================================')
print('Average Precision, AP@%d: %.4f'%(nRetrieved,avg_prec_q))
print('Recall Rate@%d: \t  %.4f'%(nRetrieved,recall)) 


fig, axs = plt.subplots(2, 5, figsize=(15, 6), facecolor='w', edgecolor='w', num='Baseline')
fig.subplots_adjust(hspace = .5, wspace=.001)
fig.suptitle('Baseline', fontsize=16)
axs = axs.ravel()
#to calculate how many match image for precision = match / return
match_precision = 0
for i in range(10):
    imgfile = os.path.join(dbpath, str(nearest_idx[i+1]) + '.jpg')
    matched_img = cv2.cvtColor(cv2.imread(imgfile), cv2.COLOR_BGR2RGB)
    axs[i].imshow(matched_img)
    axs[i].set_title(str(str(i+1) + '. ' + str(labels[retrievedCats[i]])))
    axs[i].set_xticks([])
    axs[i].set_yticks([])
  
for i in range(500):
	if(labels[gt_idx] == labels[retrievedCurve[i]]):
		match_precision = match_precision + 1
	temp_return = i + 1
	temp = match_precision / temp_return
	precision_base.append(temp)
	temp_recall = match_precision/50   #50 is number of relevant image # each query has 50 images
	recall_base.append(temp_recall)

#print(precision_base)
#print("")
#print(recall_base)
print("")


# *** Precision-Recall Curve-----------------------------------------------

print("")	
recall_array = np.linspace(0.1, 1, 10)

plt.figure("Comparison of Precision-Recall Curve")
plt.suptitle("Comparison of Precision-Recall Curve")
plt.plot(recall_bow, precision_bow, label='BoW')
plt.plot(recall_tfidf, precision_tfidf, label='TF-IDF')
plt.plot(recall_base, precision_base, label='Baseline')
plt.axis([0, 1.05, 0, 1.05])
plt.xlabel('recall')
plt.ylabel('precision')
plt.legend()
plt.show()


