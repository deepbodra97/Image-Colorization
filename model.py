from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, UpSampling2D, InputLayer, BatchNormalization
from keras.callbacks import TensorBoard

import cv2
import numpy as np

from utils import rgb_to_lab, lab_to_rgb

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # disable GPU

dataset_path = "./data/face_images/"
augmented_path = "./data/augmented_data/"
models_path = "./models/"

def load_dataset(dataset_path):
	X = []
	for filename in os.listdir(dataset_path):
		# img = rgb_to_lab(cv2.imread(dataset_path+filename))
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

def augment(batches):
	batch_num = 0
	while True:
		batch = next(batches)
		augmented_batch = np.zeros((batch.shape[0], 128, 128, 3))
		for i, image in enumerate(batch):
			scalar = np.random.uniform(0.6, 1.0)
			image = image*scalar
			cv2.imwrite(augmented_path+str(batch_num)+"_"+str(i)+".jpg", image)
			augmented_batch[i] = rgb_to_lab(image.astype(np.uint8))
		batch_num += 1
		X_batch = augmented_batch[:,:,:,:1]
		Y_batch = augmented_batch[:,:,:,1:]
		yield (X_batch, Y_batch)


def define_model():
	model = Sequential()
	model.add(InputLayer(input_shape=(128, 128, 1)))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
	model.add(BatchNormalization())
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
	model.add(BatchNormalization())
	model.add(UpSampling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization())
	model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
	model.add(UpSampling2D((2, 2)))
	model.compile(optimizer='rmsprop', loss='mse')
	model.summary()
	return model

X = load_dataset(dataset_path)
train, test = train_test_split(X, 0.9)

batch_size = 5
train_batches = datagen.flow(train, batch_size=batch_size)
augmented_batches = augment(train_batches)

model = define_model()
# model.fit(image_a_b_gen(batch_size), callbacks=[tensorboard], epochs=1, steps_per_epoch=10)
model.fit(augmented_batches, epochs=50, steps_per_epoch=len(train)//batch_size) # augmented size = steps per epoch * batch size
model.save('./models/model.h5')

test_lab = np.zeros((test.shape[0], 128, 128, 3))
print('test shape', test.shape)
for i, image in enumerate(test):
	test_lab[i] = rgb_to_lab(image)
Xtest_lab = test_lab[:,:,:,:1]
Ytest_lab = test_lab[:,:,:,1:]
print(model.evaluate(Xtest_lab, Ytest_lab))

# Test model
output = model.predict(Xtest_lab)

# # Output colorizations
for i in range(len(output)):
	cur = np.zeros((128, 128, 3))
	cur[:,:,0] = Xtest_lab[i][:,:,0]
	cur[:,:,1:] = output[i]
	cv2.imwrite("./results/img_"+str(i)+".jpg", lab_to_rgb(cur))