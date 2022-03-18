import math
import sys

import cv2
import numpy as np


class ImageInfo:
    def __init__(self, name, img, position):
        self.name = name
        self.img = img
        self.position = position


def imageBoundingBox(img, M):
    """
       This is a useful helper function that you might choose to implement
       that takes an image, and a transform, and computes the bounding box
       of the transformed image.

       INPUT:
         img: image to get the bounding box of
         M: the transformation to apply to the img
       OUTPUT:
         minX: int for the minimum X value of a corner
         minY: int for the minimum Y value of a corner
         maxX: int for the maximum X value of a corner
         maxY: int for the maximum Y value of a corner
    """
    #TODO 8
    #TODO-BLOCK-BEGIN

    height, width = img.shape[0], img.shape[1]

    corner1 = [0,0,1]
    corner2 = [0, height-1, 1]
    corner3 = [width-1, 0, 1]
    corner4 = [width-1, height-1, 1]

    transformed_corner1 = np.dot(M, corner1)
    transformed_corner2 = np.dot(M, corner2)
    transformed_corner3 = np.dot(M, corner3)
    transformed_corner4 = np.dot(M, corner4)

    homo_c1 = transformed_corner1 / transformed_corner1[2]
    homo_c2 = transformed_corner2 / transformed_corner2[2]
    homo_c3 = transformed_corner3 / transformed_corner3[2]
    homo_c4 = transformed_corner4 / transformed_corner4[2]

    transformed_corners = np.zeros((4,3))

    transformed_corners[0] = homo_c1
    transformed_corners[1] = homo_c2
    transformed_corners[2] = homo_c3
    transformed_corners[3] = homo_c4

    minX = np.amin(transformed_corners[:,0])
    maxX = np.amax(transformed_corners[:,0])
    minY = np.amin(transformed_corners[:,1])
    maxY = np.amax(transformed_corners[:,1])

    # raise Exception("TODO in blend.py not implemented")
    # TODO-BLOCK-END
    return int(minX), int(minY), int(maxX), int(maxY)


def accumulateBlend(img, acc, M, blendWidth):
    """
       INPUT:
         img: image to add to the accumulator
         acc: portion of the accumulated image where img should be added
         M: the transformation mapping the input image to the accumulator
         blendWidth: width of blending function. horizontal hat function
       OUTPUT:
         modify acc with weighted copy of img added where the first
         three channels of acc record the weighted sum of the pixel colors
         and the fourth channel of acc records a sum of the weights
    """
    # BEGIN TODO 10
    # Fill in this routine
    #TODO-BLOCK-BEGIN

    minX, minY, maxX, maxY = imageBoundingBox(img, M)
    height, width = img.shape[0], img.shape[1]

    print(minX, minY, maxX, maxY)
    print(height, width)
    for i in range(minX, maxX):
        for j in range(minY, maxY):
            inv_warp_pixel = np.array([i, j, 1.0])
            inv_warp_pixel = np.dot(np.linalg.inv(M), inv_warp_pixel)
            norm_x = min(inv_warp_pixel[0] / inv_warp_pixel[2], width-1) #resampled from original and normalized
            norm_y = min(inv_warp_pixel[1] / inv_warp_pixel[2], height-1)
            
            r = img[int(norm_y), int(norm_x), 0] 
            g = img[int(norm_y), int(norm_x), 1]
            b = img[int(norm_y), int(norm_x), 2]
            if r == 0 and g == 0 and b == 0:
                continue  
            color = np.array([0, 0, 0])
            flr_x = max(math.floor(norm_x), 0)
            ceil_x = min(math.ceil(norm_x), width-1)
            flr_y = max(math.floor(norm_y), 0)
            ceil_y = min(math.ceil(norm_y), height-1)
            x2 = ceil_x - norm_x 
            x1 = norm_x - flr_x
            y2 = ceil_y - norm_y 
            y1 = norm_y - flr_y
            for k in range(3):
                color[k] = img[int(norm_y),int(norm_x),k]
                c11 = img[flr_y, flr_x, k] 
                c12 = img[flr_y, ceil_x, k]
                c21 = img[ceil_y, flr_x, k]
                c22 = img[ceil_y, ceil_x, k]
                if x2 == 0 or x1 == 0 or y2 == 0 or y1 == 0:
                    continue
                new_color = (((c11 * x2) + (c12 * x1)) * y2) + (((c21 * x2) + (c22 * x1)) * y1)
                color[k] = new_color

            # blend with neighbour using distance map
            if norm_x >=0 and norm_x < width-1 and norm_y >=0 and norm_y < height-1:
                weight = 1.0 # the default for anything that's not in the blended width
                if (norm_x < 0 + blendWidth) and (norm_x >= 0):
                    weight = 1.0 * norm_x/blendWidth
                if (norm_x > width - blendWidth) and (norm_x <= width):
                    weight = 1.0 * (width - norm_x)/blendWidth
            
                acc[j,i,3] = acc[j,i,3] + weight #keep adding weights to the 4th
                acc[j,i,0] = acc[j,i,0] + (color[0]*weight)
                acc[j,i,1] = acc[j,i,1] + (color[1]*weight)     
                acc[j,i,2] = acc[j,i,2] + (color[2]*weight)

    return acc

    # raise Exception("TODO in blend.py not implemented")
    #TODO-BLOCK-END
    # END TODO


def normalizeBlend(acc):
    """
       INPUT:
         acc: input image whose alpha channel (4th channel) contains
         normalizing weight values
       OUTPUT:
         img: image with r,g,b values of acc normalized
    """
    # BEGIN TODO 11
    # fill in this routine..
    #TODO-BLOCK-BEGIN

    height = acc.shape[0]
    width = acc.shape[1]
    img = np.zeros((height, width, 3))
    for i in range(0, width):
        for j in range(0, height):
            if acc[j,i,3] > 0:
                img[j,i,0] = int(acc[j,i,0] / acc[j,i,3])
                img[j,i,1] = int(acc[j,i,1] / acc[j,i,3])
                img[j,i,2] = int(acc[j,i,2] / acc[j,i,3])
            else:
                img[j,i,0] = 0
                img[j,i,1] = 0
                img[j,i,2] = 0
    img = np.uint8(img)

    #TODO-BLOCK-END
    # END TODO
    return img


def getAccSize(ipv):
    """
       This function takes a list of ImageInfo objects consisting of images and
       corresponding transforms and Returns useful information about the accumulated
       image.

       INPUT:
         ipv: list of ImageInfo objects consisting of image (ImageInfo.img) and transform(image (ImageInfo.position))
       OUTPUT:
         accWidth: Width of accumulator image(minimum width such that all tranformed images lie within acc)
         accWidth: Height of accumulator image(minimum height such that all tranformed images lie within acc)

         channels: Number of channels in the accumulator image
         width: Width of each image(assumption: all input images have same width)
         translation: transformation matrix so that top-left corner of accumulator image is origin
    """

    # Compute bounding box for the mosaic
    minX = np.Inf
    minY = np.Inf
    maxX = 0
    maxY = 0
    channels = -1
    width = -1  # Assumes all images are the same width
    M = np.identity(3)
    for i in ipv:
        M = i.position
        img = i.img
        _, w, c = img.shape
        if channels == -1:
            channels = c
            width = w

        # BEGIN TODO 9
        # add some code here to update minX, ..., maxY
        #TODO-BLOCK-BEGIN
        temp_minX, temp_minY, temp_maxX, temp_maxY = imageBoundingBox(img, M)
        
        if temp_minX < minX:
            minX = temp_minX
        if temp_minY < minY:
            minY = temp_minY
        if temp_maxX > maxX:
            maxX = temp_maxX
        if temp_maxY > maxY:
            maxY = temp_maxY
        
        # raise Exception("TODO in blend.py not implemented")
        #TODO-BLOCK-END
        # END TODO

    # Create an accumulator image
    accWidth = int(math.ceil(maxX) - math.floor(minX))
    accHeight = int(math.ceil(maxY) - math.floor(minY))
    print('accWidth, accHeight:', (accWidth, accHeight))
    translation = np.array([[1, 0, -minX], [0, 1, -minY], [0, 0, 1]])

    return accWidth, accHeight, channels, width, translation


def pasteImages(ipv, translation, blendWidth, accWidth, accHeight, channels):
    acc = np.zeros((accHeight, accWidth, channels + 1))
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        M = i.position
        img = i.img

        M_trans = translation.dot(M)
        accumulateBlend(img, acc, M_trans, blendWidth)

    return acc


def getDriftParams(ipv, translation, width):
    # Add in all the images
    M = np.identity(3)
    for count, i in enumerate(ipv):
        if count != 0 and count != (len(ipv) - 1):
            continue

        M = i.position

        M_trans = translation.dot(M)

        p = np.array([0.5 * width, 0, 1])
        p = M_trans.dot(p)

        # First image
        if count == 0:
            x_init, y_init = p[:2] / p[2]
        # Last image
        if count == (len(ipv) - 1):
            x_final, y_final = p[:2] / p[2]

    return x_init, y_init, x_final, y_final


def computeDrift(x_init, y_init, x_final, y_final, width):
    A = np.identity(3)
    drift = (float)(y_final - y_init)
    # We implicitly multiply by -1 if the order of the images is swapped...
    length = (float)(x_final - x_init)
    A[0, 2] = -0.5 * width
    # Negative because positive y points downwards
    A[1, 0] = -drift / length

    return A


def blendImages(ipv, blendWidth, is360=False, A_out=None):
    """
       INPUT:
         ipv: list of input images and their relative positions in the mosaic
         blendWidth: width of the blending function
       OUTPUT:
         croppedImage: final mosaic created by blending all images and
         correcting for any vertical drift
    """
    accWidth, accHeight, channels, width, translation = getAccSize(ipv)
    acc = pasteImages(
        ipv, translation, blendWidth, accWidth, accHeight, channels
    )
    compImage = normalizeBlend(acc)

    # Determine the final image width
    outputWidth = (accWidth - width) if is360 else accWidth
    x_init, y_init, x_final, y_final = getDriftParams(ipv, translation, width)
    # Compute the affine transform
    A = np.identity(3)
    # BEGIN TODO 12
    # fill in appropriate entries in A to trim the left edge and
    # to take out the vertical drift if this is a 360 panorama
    # (i.e. is360 is true)
    # Shift it left by the correct amount
    # Then handle the vertical drift
    # Note: warpPerspective does forward mapping which means A is an affine
    # transform that maps accumulator coordinates to final panorama coordinates
    #TODO-BLOCK-BEGIN
    # raise Exception("TODO in blend.py not implemented")

    if is360:
        A = computeDrift(x_init, y_init, x_final, y_final, width)
       
    #TODO-BLOCK-END
    # END TODO

    if A_out is not None:
        A_out[:] = A

    # Warp and crop the composite
    croppedImage = cv2.warpPerspective(
        compImage, A, (outputWidth, accHeight), flags=cv2.INTER_LINEAR
    )

    return croppedImage