import numpy as np
import os
import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tkinter as tk
from PIL import Image, ImageDraw
from imutils.contours import sort_contours
import argparse
import imutils
import cv2

import fct_utiles as fct


def load_az_dataset(datasetPath):

	# initialize the list of data and labels
	data = []
	labels = []

	# loop over the rows of the A-Z handwritten digit dataset
	for row in open(datasetPath):

		# parse the label and image from the row
		row = row.split(",")
		label = int(row[0])
		image = np.array([int(x) for x in row[1:]], dtype="uint8")

		# images are represented as single channel (grayscale) images
		# that are 28x28=784 pixels -- we need to take this flattened
		# 784-d list of numbers and repshape them into a 28x28 matrix
		# image = image.reshape((28, 28))

		# update the list of data and labels
		data.append(image)
		labels.append(label)

	# convert the data and labels to NumPy arrays
	data = np.array(data, dtype="float32")
	labels = np.array(labels, dtype="int")

	# return a 2-tuple of the A-Z data and labels
	return (data, labels)


def train_model (datasetPath, modelName):

	modelPath = fct.MODELS_LOCAL_PATH + modelName + '.h5'

	# Split to train and test
	data, labels = load_az_dataset(datasetPath)
	x = data
	y = labels
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=5)

	# Change depth of image to 1
	x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
	x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

	# Change type from int to float and normalize to [0, 1]
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255

	# Optionally check the number of samples
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	# Convert class vectors to binary class matrices (transform the problem to multi-class classification)
	num_classes = len(fct.alphabet)
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	# Check if there is a pre-trained model
	if not os.path.exists(modelPath):
		# Create a neural network with 2 convolutional layers and 2 dense layers
		model = Sequential()
		model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(28,28,1)))
		model.add(Convolution2D(32, 3, 3, activation='relu'))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.25))
		
		model.add(Flatten())
		model.add(Dense(128, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(num_classes, activation='softmax'))

		model.summary()
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

		# Train the model
		model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(x_test, y_test))

		# Save the model
		model.save(modelPath)

	else:
		# Load the model from disk
		model = load_model(modelPath)

	# Get loss and accuracy on validation set
	score = model.evaluate(x_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

	return modelPath


def computer_metrics (modelPath):
	datasetPath = fct.DATASET_PATH

	# Load the model
	model = load_model(modelPath)

	# Split to train and test
	data, labels = load_az_dataset(datasetPath)
	x = data
	y = labels
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=5)

	# Process the images as in training
	x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
	x_test = x_test.astype('float32')
	x_test /= 255

	# Make predictions
	predictions = model.predict_classes(x_test, verbose=0)
	correct_indices = np.nonzero(predictions == y_test)[0]
	incorrect_indices = np.nonzero(predictions != y_test)[0]

	# Optionally plot some images
	print("Correct: %d" %len(correct_indices))
	plt.figure()
	for i, correct in enumerate(correct_indices[:9]):
		plt.subplot(3,3,i+1)
		plt.tight_layout()
		plt.imshow(x_test[correct].reshape(28,28), cmap='gray', interpolation='none')
		plt.title("Predicted {}, Class {}".format(predictions[correct], y_test[correct]))
	
	print("Incorrect: %d" %len(incorrect_indices))
	plt.figure()
	for i, incorrect in enumerate(incorrect_indices[:9]):
		plt.subplot(3,3,i+1)
		plt.tight_layout()
		plt.imshow(x_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
		plt.title("Predicted {}, Class {}".format(predictions[incorrect], y_test[incorrect]))


def drawing_test (model):

	model = load_model(model)

	def evaluate():
		img = image1.resize((28, 28)).convert('L')
		pixels = img.load()
		for i in range(img.size[0]):
			for j in range(img.size[1]):
				pixels[i,j] = 255 if pixels[i,j] == 0 else 0
		X_test = np.array(img).reshape(1, 28, 28, 1)
		predicted_classes = model.predict_classes(X_test, verbose=0)
		reslabel['text'] = "Prédiction : " + str(fct.alphabet[predicted_classes[0]])

	def clear():
		pixels = image1.load()
		for i in range(image1.size[0]):
			for j in range(image1.size[1]):
				pixels[i,j] = (255, 255,255)
		cv.delete("all")
		reslabel['text'] = "Dessine une lettre"

	def paint(event):
		x1, y1 = (event.x - 10), (event.y - 10)
		x2, y2 = (event.x + 10), (event.y + 10)
		cv.create_oval(x1, y1, x2, y2, fill="black",width=4)
		draw.ellipse([x1, y1, x2, y2],fill="black")

	root = tk.Tk()
	root.title("OCR application")
	root.resizable(False, False)

	# Create top canvas and image
	cv = tk.Canvas(root, width=280, height=280, bg='white')
	cv.pack()
	image1 = Image.new("RGB", (280, 280), (255, 255, 255))
	draw = ImageDraw.Draw(image1)
	cv.bind("<B1-Motion>", paint)

	# Create bottom label and buttons
	bottom = tk.Frame(root)
	bottom.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
	reslabel = tk.Label(text = "Dessine une lettre")
	reslabel.pack(in_=bottom, side=tk.LEFT, fill=tk.Y, expand=True)
	button = tk.Button(text="Evaluer", command=evaluate, width=6)
	button.pack(in_=bottom, side=tk.LEFT, fill=tk.Y)
	button = tk.Button(text="Effacer", command=clear, width=6)
	button.pack(in_=bottom, side=tk.RIGHT, fill=tk.Y)

	root.mainloop()


def img_test (model, img) :

	# load the handwriting OCR model
	print("[INFO] loading handwriting OCR model...")
	model = load_model(model)

	# load the input image from disk, convert it to grayscale, and blur it to reduce noise
	image = cv2.imread(img)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	# blurred = cv2.blur(gray, (5, 5))
	cleaned = remove_noise_and_smooth(blurred)
	

	# perform edge detection, find contours in the edge map, and sort the
	# resulting contours from left-to-right
	edged = cv2.Canny(cleaned, 30, 150)
	cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sort_contours(cnts, method="left-to-right")[0]

	# initialize the list of contour bounding boxes and associated
	# characters that we'll be OCR'ing
	chars = []

	#dimensions
	lettersize = 20
	padding = 4
	imgsize = 28

	# loop over the contours
	for c in cnts:
		# compute the bounding box of the contour
		(x, y, w, h) = cv2.boundingRect(c)

		# filter out bounding boxes, ensuring they are neither too small
		# nor too large
		if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):

			# extract the character and threshold it to make the character
			# appear as *white* (foreground) on a *black* background, then
			# grab the width and height of the thresholded image
			roi = gray[y:y + h, x:x + w]
			thresh = cv2.threshold(roi, 0, 255,
				cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
			(tH, tW) = thresh.shape

			# if the width is greater than the height, resize along the
			# width dimension
			if tW > tH:
				thresh = imutils.resize(thresh, width=lettersize)

			# otherwise, resize along the height
			else:
				thresh = imutils.resize(thresh, height=lettersize)

			# pad the image and force 28x28 dimensions
			padded = cv2.copyMakeBorder(thresh, top=padding, bottom=padding,
				left=padding, right=padding, borderType=cv2.BORDER_CONSTANT,
				value=(0, 0, 0))
			padded = cv2.resize(padded, (imgsize, imgsize))

			# prepare the padded image for classification via our
			# handwriting OCR model
			padded = padded.astype("float32") / 255.0
			padded = np.expand_dims(padded, axis=-1)
			
			# update our list of characters that will be OCR'd
			chars.append((padded, (x, y, w, h)))

	# extract the bounding box locations and padded characters
	boxes = [b[1] for b in chars]
	chars = np.array([c[0] for c in chars], dtype="float32")
	# OCR the characters using our handwriting recognition model
	preds = model.predict(chars)
	# define the list of label names
	labelNames = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
	labelNames = [l for l in labelNames]

	# loop over the predictions and bounding box locations together
	for (pred, (x, y, w, h)) in zip(preds, boxes):
		# find the index of the label with the largest corresponding
		# probability, then extract the probability and label
		i = np.argmax(pred)
		prob = pred[i]
		label = labelNames[i]
		# draw the prediction on the image
		print("[INFO] {} - {:.2f}%".format(label, prob * 100))
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
		cv2.putText(image, label, (x - 10, y - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
	# show the image
	cv2.imshow("Image", image)
	cv2.waitKey(0)


def remove_noise_and_smooth(img): 
    filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3) 
    kernel = np.ones((1, 1), np.uint8) 
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel) 
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel) 
    img = image_smoothening(img) 
    or_image = cv2.bitwise_or(img, closing) 
    return or_image 


def image_smoothening(img): 
    ret1, th1 = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY) 
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
    blur = cv2.GaussianBlur(th2, (1, 1), 0) 
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
    return th3 