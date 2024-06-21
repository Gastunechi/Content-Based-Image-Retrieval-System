# Content-Based-Image-Retrieval-System
This project implements a content-based image retrieval system using color and shape features to compare images. The system takes a query image as input and returns the N closest images from the database in terms of similarity.

## Objective
The goal of this project is to develop a system capable of finding the images most similar to a given image using two types of features:

Colors: Comparison of color histograms.
Shapes: Using Hu moments to compare shapes.

## Functionality
The system operates in several steps:

- Read the query image.
- Calculate the features (color histograms and Hu moments) of the query image.
For each image in the database:
- Calculate the same features.
- Calculate the distance between the features of the query image and the database image.
- Sort the database images based on the calculated distances.
- Display the N most similar images.

## Dependencies
- Python 3.x
- OpenCV
- NumPy
- OS
