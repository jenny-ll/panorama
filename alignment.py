import math
import random
import warp

import cv2
import numpy as np

eTranslate = 0
eHomography = 1


def computeHomography(f1, f2, matches, A_out=None):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        A_out -- ignore this parameter. If computeHomography is needed
                 in other TODOs, call computeHomography(f1,f2,matches)
    Output:
        H -- 2D homography (3x3 matrix)
        Takes two lists of features, f1 and f2, and a list of feature
        matches, and estimates a homography from image 1 to image 2 from the matches.
    '''
    num_matches = len(matches)

    # Dimensions of the A matrix in the homogenous linear
    # equation Ah = 0
    num_rows = 2 * num_matches
    num_cols = 9
    A_matrix_shape = (num_rows,num_cols)
    A = np.zeros(A_matrix_shape)

    for i in range(len(matches)):
        m = matches[i]
        (a_x, a_y) = f1[m.queryIdx].pt
        (b_x, b_y) = f2[m.trainIdx].pt

        #BEGIN TODO 2
        #Fill in the matrix A in this loop.
        #Access elements using square brackets. e.g. A[0,0]
        #TODO-BLOCK-BEGIN

        A[2*i,0] = a_x
        A[2*i,1] = a_y
        A[2*i,2] = 1
        A[2*i,3] = 0
        A[2*i,4] = 0
        A[2*i,5] = 0
        A[2*i,6] = -b_x*a_x
        A[2*i,7] = -b_x*a_y
        A[2*i,8] = -b_x

        A[2*i+1,0] = 0
        A[2*i+1,1] = 0
        A[2*i+1,2] = 0
        A[2*i+1,3] = a_x
        A[2*i+1,4] = a_y
        A[2*i+1,5] = 1
        A[2*i+1,6] = -b_y*a_x
        A[2*i+1,7] = -b_y*a_y
        A[2*i+1,8] = -b_y

        #TODO-BLOCK-END
        #END TODO

    U, s, Vt = np.linalg.svd(A)

    if A_out is not None:
        A_out[:] = A

    #s is a 1-D array of singular values sorted in descending order
    #U, Vt are unitary matrices
    #Rows of Vt are the eigenvectors of A^TA.
    #Columns of U are the eigenvectors of AA^T.

    #Homography to be calculated
    H = np.eye(3)

    #BEGIN TODO 3
    #Fill the homography H with the appropriate elements of the SVD
    #TODO-BLOCK-BEGIN

    H[0,0] = Vt[8,0]
    H[0,1] = Vt[8,1]
    H[0,2] = Vt[8,2]
    H[1,0] = Vt[8,3]
    H[1,1] = Vt[8,4]
    H[1,2] = Vt[8,5]
    H[2,0] = Vt[8,6]
    H[2,1] = Vt[8,7]
    H[2,2] = Vt[8,8]

    # raise Exception("TODO in alignment.py not implemented")
    #TODO-BLOCK-END
    #END TODO

    return H

def alignPair(f1, f2, matches, m, nRANSAC, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        m -- MotionModel (eTranslate, eHomography)
        nRANSAC -- number of RANSAC iterations
        RANSACthresh -- RANSAC distance threshold

    Output:
        M -- inter-image transformation matrix
        Repeat for nRANSAC iterations:
            Choose a minimal set of feature matches.
            Estimate the transformation implied by these matches
            count the number of inliers.
        For the transformation with the maximum number of inliers,
        compute the least squares motion estimate using the inliers,
        and return as a transformation matrix M.
    '''
#     eTranslate = 0
# eHomography = 1

    #BEGIN TODO 4
    #Write this entire method.  You need to handle two types of
    #motion models, pure translations (m == eTranslation) and
    #full homographies (m == eHomography).  However, you should
    #only have one outer loop to perform the RANSAC code, as
    #the use of RANSAC is almost identical for both cases.

    #Your homography handling code should call compute_homography.
    #This function should also call get_inliers and, at the end,
    #least_squares_fit.
    #TODO-BLOCK-BEGIN

    max_inliers = []
    for n in range(nRANSAC):
        if m == 1: # eHomography
            sample_matches = random.sample(matches, 4)
            h = computeHomography(f1, f2, sample_matches)
        elif m == 0:
            sample_matches = random.sample(matches, 1)
            (x, y) = f1[sample_matches[0].queryIdx].pt
            (x2, y2) = f2[sample_matches[0].trainIdx].pt
            h = np.eye(3,3)
            h[0,2] = x2-x
            h[1,2] = y2-y
        inliers = getInliers(f1, f2, matches, h, RANSACthresh)

        if len(inliers) > len(max_inliers):
            max_inliers = inliers

    M = leastSquaresFit(f1, f2, matches, m, max_inliers)
        
    # raise Exception("TODO in alignment.py not implemented")
    #TODO-BLOCK-END
    #END TODO
    return M

def getInliers(f1, f2, matches, M, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        M -- inter-image transformation matrix
        RANSACthresh -- RANSAC distance threshold

    Output:
        inlier_indices -- inlier match indices (indexes into 'matches')

        Transform the matched features in f1 by M.
        Store the match index of features in f1 for which the transformed
        feature is within Euclidean distance RANSACthresh of its match
        in f2.
        Return the array of the match indices of these features.
    '''

    inlier_indices = []

    for i in range(len(matches)):
        #BEGIN TODO 5
        #Determine if the ith matched feature f1[id1], when transformed
        #by M, is within RANSACthresh of its match in f2.
        #If so, append i to inliers
        #TODO-BLOCK-BEGIN
        (a_x, a_y) = f1[matches[i].queryIdx].pt
        (b_x, b_y) = f2[matches[i].trainIdx].pt

        vector = np.array([a_x, a_y, 1])
        transformed = np.dot(M, vector)

        transformed[0] = transformed[0] / transformed[2]
        transformed[1] = transformed[1] / transformed[2]
        transformed[2] = 1

        dist = np.linalg.norm([b_x, b_y, 1]-transformed)

        if dist <= RANSACthresh:
            inlier_indices.append(i)

        # raise Exception("TODO in alignment.py not implemented")
        #TODO-BLOCK-END
        #END TODO

    return inlier_indices

def leastSquaresFit(f1, f2, matches, m, inlier_indices):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        m -- MotionModel (eTranslate, eHomography)
        inlier_indices -- inlier match indices (indexes into 'matches')

    Output:
        M - transformation matrix

        Compute the transformation matrix from f1 to f2 using only the
        inliers and return it.
    '''

    # This function needs to handle two possible motion models,
    # pure translations (eTranslate)
    # and full homographies (eHomography).

    M = np.eye(3)

    if m == eTranslate:
        #For spherically warped images, the transformation is a
        #translation and only has two degrees of freedom.
        #Therefore, we simply compute the average translation vector
        #between the feature in f1 and its match in f2 for all inliers.

        u = 0.0
        v = 0.0

        for i in range(len(inlier_indices)):
            #BEGIN TODO 6
            #Use this loop to compute the average translation vector
            #over all inliers.
            #TODO-BLOCK-BEGIN
            (a_x, a_y) = f1[matches[inlier_indices[i]].queryIdx].pt
            (b_x, b_y) = f2[matches[inlier_indices[i]].trainIdx].pt
            u = u + (b_x - a_x)
            v = v + (b_y - a_y)
            # raise Exception("TODO in alignment.py not implemented")
            #TODO-BLOCK-END
            #END TODO

        u /= len(inlier_indices)
        v /= len(inlier_indices)

        M[0,2] = u
        M[1,2] = v

    elif m == eHomography:
        #BEGIN TODO 7
        #Compute a homography M using all inliers.
        #This should call computeHomography.
        #TODO-BLOCK-BEGIN

        inlier_matches = []
        for i in range(len(inlier_indices)):
            inlier_matches.append(matches[inlier_indices[i]])

        M = computeHomography(f1, f2, inlier_matches)
        # raise Exception("TODO in alignment.py not implemented")
        #TODO-BLOCK-END
        #END TODO

    else:
        raise Exception("Error: Invalid motion model.")

    return M

