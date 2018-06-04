# %%
import numpy as np
import os
import cv2
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda
# from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam

# %%
image_size = 128  # Pixel width and height.
min_year = 1984
max_year = 2021
year_offset = max_year - ((max_year - min_year) / 2)
files = []
labels = []
years = []

def load_year(folder, year):
	"""Load the data for a single letter label."""
	image_files = os.listdir(folder)
	# stupid file names...
	thumbs = list(filter(lambda f: 'jpgtn.jpg' in f, image_files))
	files.extend(map(lambda f: os.path.join(folder, f), thumbs));
	labels.extend([year] * len(thumbs));
	# labels.extend([(year - min_year) / (max_year - min_year) * 2 - 1] * len(thumbs));
	years.extend([year] * len(thumbs))
	#print('%s images for year %s' % (len(thumbs), year))

def load_all(folder):
	files = []
	labels = []
	print(folder)
	for dir in os.listdir(folder):
		if dir.isdigit():
			year = int(dir)
            # Load the data for a single year label.
			load_year(os.path.join(folder, dir), year);
		else:
			print('Skipping file/folder ', dir)

load_all('C:\dev\Posters')

plt.hist(years, 35)
plt.show()
plt.hist(labels)
plt.show()

print(len(files))
print(len(labels))


# %%
def load_train():
    X_train = []
    y_train = labels
    print('Read train images')
    for image_path in files:
        # image_path = image_path.replace('\\', '/')
        #read the data from the file
        with open(image_path, 'rb') as infile:
             buf = infile.read()
        x = np.frombuffer(buf, dtype='uint8')
        # print(image_path)
        img = cv2.imdecode(x, cv2.IMREAD_COLOR);
        resized = cv2.resize(img, (image_size, image_size)).astype(np.float32)
        # resized = resized.transpose((2,0,1))
        X_train.append(resized)
    return X_train, y_train

def read_and_normalize_train_data():
    train_data, train_target = load_train()
    train_data = np.array(train_data, dtype=np.float32)
    train_target = np.array(train_target, dtype=np.float32)
    m = train_data.mean()
    s = train_data.std()

    print ('Train mean, sd:', m, s )
    train_data -= m
    train_data /= s
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target

# num_samples = 1999
train_data, train_target = read_and_normalize_train_data()
# train_data = train_data[0:num_samples,:,:,:]
# train_target = train_target[0:num_samples]
X_train, X_valid, y_train, y_valid = train_test_split(train_data, train_target, test_size=0.2)

# %%
def create_model():
    model = Sequential()
    model.add(Conv2D(32, 3, padding="same", input_shape=(image_size, image_size, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, 3, padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, 5, padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(1024))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('linear'))
    model.add(Lambda(lambda x: x + year_offset))

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001))
    return model

model = create_model()


# %%
model.fit(X_train, y_train, batch_size=50, nb_epoch=10, verbose=1, validation_data=(X_valid, y_valid) )


# %%
cv_size = 499
predictions_valid = model.predict(X_valid, batch_size=50, verbose=1)
compare = pd.DataFrame(data={'original':y_valid.reshape((cv_size,)),
         'prediction':predictions_valid.reshape((cv_size,))})
compare.to_csv('compare.csv')
