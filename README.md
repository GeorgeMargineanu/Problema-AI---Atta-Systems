# Problema-AI---Atta-Systems
https://www.dropbox.com/s/69l2qg0s0uuljf0/Problema%20AI.zip?dl=0&amp;file_subpath=%2FProblema+AI

This algorithm aims to segment images from CT scan images.
I approached this problem using classic methods for preprocessing; I clustered the pixels using KMeans and tried to get an image containing the area of interest.
Then, I used this image as a mask and convolved it with the initial image. For this result I applied again KMeans to filter out the unnecessary contours.

Issues:
- For 107-HU.in the result is promising, but for the other inputs the algorithm does not segment correctly the image.
- Many function inputs were manually tested such as number of clusters and thresholds.
- In the range specified in the document soft tissue should be between 40-87, but this threshold is off, keeping almost the entire image information.

Requirements: 
1. opencv-contrib-python (NOT opencv-python)
2. matplotlib
3. numpy
4. sys

Run: py Challenge.py 107-HU.in 107-seg.in



