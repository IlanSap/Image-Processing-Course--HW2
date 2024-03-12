import cv2
import matplotlib.pyplot as plt
import numpy as np

def displayImages(src_image, target_image,title):

	# Display the original and after correction images side by side
	plt.figure(figsize=(10, 5))

	plt.subplot(1, 2, 1)
	plt.imshow(src_image, cmap='gray')
	plt.title('Original Image')

	plt.subplot(1, 2, 2)
	plt.imshow(target_image, cmap='gray')
	plt.title(title)

	plt.show()

# input: an image
# output: the image after brightness and contrast stretching correction
def BrightAndContrast(image):
	min_intensity = 50
	max_intensity = 200

	# Perform brightness and contrast stretching
	stretched_image = np.clip(
		(image - image.min()) / (image.max() - image.min()) * (max_intensity - min_intensity) + min_intensity, 0,
		255).astype(np.uint8)

	displayImages(image, stretched_image, 'Stretched Image')


# input: an image
# output: the image after histogram equalization correction
def histogramEqualization(image):
	equalized_image = cv2.equalizeHist(image)

	displayImages(image, equalized_image, 'Equalized Image')


# input: an image
# outpt: the image after gamma correction
def gammaCorrection(image):
	gamma = 1.5

	# Perform gamma correction
	corrected_image = np.power(image / 255.0, gamma)
	corrected_image = (corrected_image * 255).astype(np.uint8)

	displayImages(image, corrected_image, 'Gamma Corrected Image')


def apply_fix(image):
	gammaCorrection(image)
	histogramEqualization(image)
	BrightAndContrast(image)


for i in range(1, 4):
	if(i ==1):
		path = f'{i}.png'
	else:
		path = f'{i}.jpg'

	image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

	apply_fix(image)

