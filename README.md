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

## Results
Overall, the retrieval system achieves the highest mean average precision (MAP@50) of
0.7520 value with BoW, followed with slightly lower value of 0.7518 by TF-IDF. The recall
rate@50 of BoW and TF-IDF is the same value which is 0.6200 and it is higher than the
baseline method. The full evaluation shows that BoW and TF-IDF has a better performance
than baseline method across all classes.
