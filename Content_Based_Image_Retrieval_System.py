import cv2
import numpy as np
import os

# Read input image
def read_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to load image : {image_path}")
    return image

# Calculate features: color histograms
def calculate_histogram(image):
    histogram = []
    for canal in range(3):  # For R, G, B channels
        hist = cv2.calcHist([image], [canal], None, [32], [0, 256])
        histogram.extend(hist.flatten())
    histogram = np.array(histogram)
    histogram /= histogram.sum()  # Standardization
    return histogram

# Calculate features: Hu moments for shapes
def calculate_moments_hu(image):
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    moments = cv2.moments(grey_image)
    moments_hu = cv2.HuMoments(moments).flatten()
    return moments_hu

# Calculate distance between color histograms
def distance_histogram(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)

# Calculate distance between Hu moments
def distance_moments_hu(moments1, moments2):
    return np.linalg.norm(moments1 - moments2)

# Global similarity function
def calculate_global_distance(image1, image2):
    hist1 = calculate_histogram(image1)
    hist2 = calculate_histogram(image2)
    moments1 = calculate_moments_hu(image1)
    moments2 = calculate_moments_hu(image2)

    dist_color = distance_histogram(hist1, hist2)
    dist_form = distance_moments_hu(moments1, moments2)
    
    omega1 = 0.5
    omega2 = 0.5
    total_distance = omega1 * dist_color + omega2 * dist_form
    return total_distance

# Checks if file is an image
def is_an_image(file):
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.bmp')
    return file.lower().endswith(valid_extensions)

# Search for most similar images
def search_similar_images(query_path, images_folder, N=5):
    image_requete = read_image(query_path)
    distances = []

    for file in os.listdir(images_folder):
        if is_an_image(file):
            image_path = os.path.join(images_folder, file)
            if os.path.isfile(image_path):
                try:
                    image_base = read_image(image_path)
                    distance = calculate_global_distance(image_requete, image_base)
                    distances.append((file, distance))
                except ValueError as e:
                    print(e)

    distances = sorted(distances, key=lambda x: x[1])
    return distances[:N]

# Example of use
results = search_similar_images('coil-100/coil-100/obj1__200.png', 'coil-100/coil-100', N=10) # Specify the file path of the query image and the folder path in which to search for similar images
for file, distance in results:
    print(f'Image: {file}, Distance: {distance}')
