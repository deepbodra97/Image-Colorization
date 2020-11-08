from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, InputLayer, BatchNormalization, Dense
from keras.models import load_model

import cv2
import numpy as np

from utils import rgb_to_lab, lab_to_rgb

import os

dataset_path = "./data/face_images/"
augmented_path = "./data/augmented_data/"
models_path = "./models/"
results_path = "./results/"

def load_dataset(dataset_path):
	X = []
	for filename in os.listdir(dataset_path):
		img = cv2.imread(dataset_path+filename)
		X.append(img)
	return np.array(X)

def train_test_split(X, split):
	split_idx = int(split*len(X))
	return X[:split_idx], X[split_idx:]

datagen = ImageDataGenerator(
			zoom_range=0.2,
			horizontal_flip=True
		)

def augment(batches, activation_last):
	batch_num = 0
	while True:
		batch = next(batches)
		X_batch = np.zeros((batch.shape[0], 128, 128, 1))
		Y_batch = np.zeros((batch.shape[0], 2, 1))
		for i, image in enumerate(batch):
			scalar = np.random.uniform(0.6, 1.0)
			image = image*scalar
			cv2.imwrite(augmented_path+str(batch_num)+"_"+str(i)+".jpg", image)
			lab_image = rgb_to_lab(image.astype(np.uint8))
			l, a, b = lab_image[:,:,:1], lab_image[:,:,1:2], lab_image[:,:,2:3]
			X_batch[i] = l
			Y_batch[i,0] = np.mean(a)
			Y_batch[i,1] = np.mean(b)
		batch_num += 1
		yield (X_batch, Y_batch)


def define_model(activation_last):
	model = Sequential()
	model.add(InputLayer(input_shape=(128, 128, 1)))
	model.add(Conv2D(3, (2, 2), activation='relu', padding='same'))
	model.add(MaxPooling2D((2,2)))
	model.add(Conv2D(3, (2, 2), activation='relu', padding='same'))
	model.add(MaxPooling2D((2,2)))
	model.add(Conv2D(3, (2, 2), activation='relu', padding='same'))
	model.add(MaxPooling2D((2,2)))
	model.add(Conv2D(3, (2, 2), activation='relu', padding='same'))
	model.add(MaxPooling2D((2,2)))
	model.add(Conv2D(3, (2, 2), activation='relu', padding='same'))
	model.add(MaxPooling2D((2,2)))
	model.add(Conv2D(3, (2, 2), activation='relu', padding='same'))
	model.add(MaxPooling2D((2,2)))
	model.add(Conv2D(2, (2, 2), activation='relu', padding='same'))
	model.add(MaxPooling2D((2,2)))	
	model.summary()
	model.compile(optimizer='rmsprop', loss='mse')
	return model

def predict(x, dir_name):
	output = model.predict(x)
	for i in range(len(output)):
		print(output)
		# cur = np.zeros((128, 128, 3))
		# cur[:,:,0] = x[i][:,:,0]
		# cur[:,:,1:] = output[i]
		# cv2.imwrite(results_path+activation_last+"/"+dir_name+"/"+str(i)+".jpg", lab_to_rgb(cur))

def get_results():
	test_lab = np.zeros((test.shape[0], 128, 128, 1))
	train_lab = np.zeros((test.shape[0], 128, 128, 1))
	
	for i, image in enumerate(test):
		train_lab[i] = rgb_to_lab(image)

	for i, image in enumerate(test):
		test_lab[i] = rgb_to_lab(image)

	Xtrain_lab = train_lab[:,:,:,:1]
	Ytrain_lab = train_lab[:,:,:,1:]

	Xtest_lab = test_lab[:,:,:,:1]
	Ytest_lab = test_lab[:,:,:,1:]
	
	print("Loss on test set of 75 images")
	print(model.evaluate(Xtest_lab, Ytest_lab))
	predict(Xtest_lab, 'test')
	predict(Xtrain_lab, 'train')

# main
print("Choose an Activation Function for last layer\n1. Relu\n2. Tanh\n")
activation_last = {'1': 'relu', '2': 'tanh'}[input()]

X = load_dataset(dataset_path)
train, test = train_test_split(X, 0.9)

if not os.path.exists(models_path+'regressor.h5'):
	batch_size = 5
	train_batches = datagen.flow(train, batch_size=batch_size)
	augmented_batches = augment(train_batches, activation_last)

	model = define_model(activation_last)
	model.fit(augmented_batches, epochs=10, steps_per_epoch=len(train)//batch_size) # augmented size = steps per epoch * batch size
	model.save(models_path+'regressor.h5')
	get_results()
else:
	model = load_model(models_path+'regressor.h5')
	get_results()