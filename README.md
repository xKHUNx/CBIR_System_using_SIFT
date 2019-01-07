# Content-based Image Retireval System using SIFT
Content-based Image Retireval System using SIFT. An image retrieval system that applies SIFT and K-mean clustering for feature extraction. Different visual word representations are tested. The system is built based on MalayaKew Dataset.

![alt text](https://github.com/xKHUNx/CBIR_System_using_SIFT/blob/master/image_retrieval.PNG)

## Introduction
In this project, we had built an image retrieval system that retrieves a set number of
image of leaves, given an image as query. This image retrieval system is built on the
MalayaKew (MK) Leaf dataset, which consist of the images of the leaves of 10 species
classes of plant. For the retrieval system, we have 50 images of the leaves for each species.
So a total of 500 images is in the retrieval system.
<br>
The retrieval system first extract the visual features of all the leaves in the dataset, using SIFT
(scale-invariant feature transform). The features are then clustered using K-means algorithm
to produce a set of visual words. The feature representation of the images, including that of
the query image, are then compared using cosine similarity and, the top 10 most similar
images are retrieved.
<br>
We’ve compared the performance of the image retrieval system using different settings: (1)
SIFT features with bag-of-word, (2) SIFT features with TF-IDF weighting (3) RGB
histogram, which serves as the baseline method.
In general, we found that using bag-of-word and TF-IDF method outperform the baseline
method. Both TF-IDF and bag-of-word had more or less similar performance, although
sometimes one of them will slightly outperform the others.

## Description of Methods Used
### Feature Extraction
We first extract the features of each image using SIFT. We set the contrast threshold to 0, to
capture as many key points as possible. Each image will have different number of keypoints
and thus its descriptor. We save only the descriptors from all images, which represent the
local features of the key points of each image.
### Generating Visual Vocabulary
To generate the visual vocabulary, we perform clustering on the local features of all images
using K-mean. After some experiment, we settle on K=50. This means we will have 50 visual
vocabulary or word, these words will be used as the global feature for our calculating the
similarity of the images.
<br>
We save the K-mean model, so we can map the features extracted from SIFT to one of the 50
visual words. The model works for images in the database, as well as query image, which are
not in the database. More details on Image Retrieval section.
### Feature Representation
We set out to test different way of representing the features: (1) Bag-of-words, (2) TF-IDF
(term frequency–inverse document frequency).
Bag-of-words is the represented as the count of the occurences of each visual word that are
found in the image.
<br>
TF-IDF is used to compensate for the uneven occurrence of visual words across all images. It
apply weighting to the visual words based on its term frequency (TF) and its inverse
document frequency (IDF). Visual words that are rare across all images, and/or frequent in a
particular image are given a higher weightage.

### Image Retrieval
During retrieval, we first extract the features from the querying image, and we map the
features to the visual words, using the K-mean model we saved earlier. We will the obtain the
features in either bag-of-words or TF-IDF.
<br>
This step is repeated for the other 500 images. However, we will use the preloaded features of
the 500 images we saved earlier in previous step, to speed up the process.
The querying image are then compared to the 500 images using cosine similarity. The higher
the cosine similarity is, the more similar it is. The top 50 most similar images are then
retrieved, but only the top 10 are displayed, ranking from the most similar one to the least.

## Results
Overall, the retrieval system achieves the highest mean average precision (MAP@50) of
0.7520 value with BoW, followed with slightly lower value of 0.7518 by TF-IDF. The recall
rate@50 of BoW and TF-IDF is the same value which is 0.6200 and it is higher than the
baseline method. The full evaluation shows that BoW and TF-IDF has a better performance
than baseline method across all classes.

More details [here](https://github.com/xKHUNx/CBIR_System_using_SIFT/blob/master/VIP%20Assignment%202%20Report.pdf)
