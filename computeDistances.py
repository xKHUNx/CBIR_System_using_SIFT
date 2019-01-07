"""
computeDistances.py

YOUR WORKING FUNCTION for computing pairwise distances between features

"""
from scipy.spatial import distance

# you are allowed to import other Python packages above
##########################
def computeDistances(fv):
    # Inputs
    # fv: A N-by-D array containing D-dimensional feature vector of 
    #     N number of data (images)
    # 
    # Output
    # D: N-by-N square matrix containing the pairwise distances between
    #    all samples, i.e. the first row shows the distance
    #    between the first sample and all other samples 
    #    (columns)
    #
    #########################################################################
    # ADD YOUR CODE BELOW THIS LINE
    
    # This is the baseline distance measure: Euclidean (L2) distance
    D = distance.squareform(distance.pdist(fv, 'cosine') )
    
        
    # END OF YOUR CODE
    #########################################################################
    return D