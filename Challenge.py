# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 17:58:39 2022

@author: George
"""

from sklearn.cluster import KMeans
import cv2
import matplotlib.pyplot as plt
import numpy as np


def read_image(path):
    """
    @path: string

    Tries to read the image from path.
    """
    try:
        image = []
        with open(path) as fisier:
            score = fisier.read()  # Read all file in case values are not on a single line
            image.append([int(x) for x in score.split()])

        image = np.array(image)
        image = np.reshape(image, (512, 512))
        return image
    except:
        raise Exception("Something went wrong when reading from file.")


def compute_salience(image):
    """
    @image: numpy.array

    This static saliency detector operates on the log-spectrum of an image, 
    computes saliency residuals in this spectrum, and then maps the corresponding
    salient locations back to the spatial domain.
    """
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    (success, saliencyMap) = saliency.computeSaliency(image)
    return (saliencyMap * 255).astype("uint8")


def revert_white_black(image):
    """
    @image: numpy.array

    Takes a binary input image.
    Reverts image from 0-255 to 255-0.
    """
    def function(x): return 0 if x == 255 else 255
    applyall = np.vectorize(function)
    return applyall(image).astype(np.uint8)


def keep_only_soft_tissue(image):
    """
    @image: numpy.array

    Takes a grayscale input image and applies a manual threshold to keep only 
    the soft tissue, in this case between 70 and 144.
    """
    def function(x): return 0 if not (x < 144 and x > 70) else x
    applyall = np.vectorize(function)
    return applyall(image).astype(np.uint8)


def keep_one_cluster(image, label):
    """
    @image: numpy.array
    @label: int

    Takes a grayscale input image with pixel values from 0 to number of classes
    (calculated in KMeans) and the label.
    The output is an image containing the target class.
    """
    def function(x): return 0 if x != label else x
    applyall = np.vectorize(function)
    return applyall(image).astype(np.uint8)


def flatten_image(image):
    """
    @image: numpy.array

    Takes a grayscale image and returns a column numpy array of size
    image.width * image.height
    """
    image = np.reshape(image, (image.shape[0] * image.shape[1]))
    return image.reshape((-1, 1))


def plot2figures(image_1, image_2):
    """
    @image_1: numpy.array
    @image_2: numpy.array

    Takes 2 input images to be displayed side by side.
    """

    f = plt.figure()
    f.add_subplot(1, 2, 1)
    plt.imshow(image_1, cmap='gray')

    f.add_subplot(1, 2, 2)
    plt.imshow(image_2, cmap='gray')


def first_processing(image):
    """
    @image: numpy.array

    Process raw image.
    """
    # keep the area containing the target shape.
    image = keep_only_soft_tissue(image)

    # flatten the image
    image = flatten_image(image)

    # apply Kmeans and keep 1 cluster
    kmeans = KMeans(n_clusters=5, random_state=0).fit(image)
    labels = kmeans.labels_.astype(np.uint8)
    km = np.reshape(labels, (512, 512))

    # enhance contrast with cv2.equalizeHist
    filtered = cv2.equalizeHist(km)

    # apply threshold
    th, im_gray_th_otsu = cv2.threshold(filtered, 128, 192, cv2.THRESH_OTSU)

    # compute saliency
    saliencyMap = compute_salience(im_gray_th_otsu)
    ret, threshMap = cv2.threshold(
        (saliencyMap * 255).astype('uint8'), 155, 255, cv2.THRESH_BINARY)

    # reverse 0-255
    threshMap = revert_white_black(threshMap)
    return threshMap


def second_processing(image):
    """
    @image: numpy.array
    """
    # flatten the image
    image = flatten_image(image)

    # KMeans with less clusters
    kmeans = KMeans(n_clusters=3, random_state=0).fit(image)
    labels = kmeans.labels_.astype(np.uint8)
    labels = np.reshape(labels, (512, 512))

    # keep only the cluster of interest
    labels = keep_one_cluster(labels, 1)
    return labels


def max_contour(image):
    """
    @image: numpy.array
    Returns the contour with the greatest area.
    """
    contours, hierarchy = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    max_area = 0

    for c in contours:
        if cv2.contourArea(c) > max_area:
            max_area = cv2.contourArea(c)
            contour = c
    return contour


def main():
    segmentation = cv2.imread(r'D:\IC3\107-opt.png')

    cale_r = r'D:\IC3\107-HU.in'

    image = read_image(cale_r)
    copy = np.array(image, copy=True)

    # output from first processing
    thresh_map = first_processing(image)

    # We use the output from first_processing as a mask for the original image.
    # This way the image becomes the area of interest.
    processed = (copy * thresh_map).astype(np.uint8)

    # output from second_processing
    labels = second_processing(processed)

    # contour variable the contour with greatest area
    contour = max_contour(labels)

    # in order to use the cv2.drawContours, the source image needs to have 3 channels.
    # Using cv2.COLOR_GRAY2RGB the image is turned to RGB but unchanged visually.
    gray_to_rgb = cv2.cvtColor(labels, cv2.COLOR_GRAY2RGB)

    output = cv2.drawContours(gray_to_rgb, contour, -1, (0, 255, 0), 3)

    # plots the output
    plot2figures(output, segmentation)


if __name__ == "__main__":
    main()
