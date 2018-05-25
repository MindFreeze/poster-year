# %%
import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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
	"""Load the data for a single letter label."""
	dirs = os.listdir(folder)
	print(folder)
	for dir in os.listdir(folder):
		if dir.isdigit():
			year = int(dir)
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
data_train, data_test, labels_train, labels_test = train_test_split(files, labels, test_size=0.2)

# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename, label):
    image_decoded = tf.image.decode_jpeg(tf.read_file(filename), channels=3)
    image_resized = tf.image.resize_images(image_decoded, [image_size, image_size])
    # return image_resized, label
    return {"image_data": image_resized, "title": filename}, tf.to_float(label)

def train_input_fn():
    """An input function for training"""
    dataset = tf.data.Dataset.from_tensor_slices((tf.constant(data_train), tf.constant(labels_train)))
    dataset = dataset.map(_parse_function)
    return dataset.shuffle(1000).batch(256).repeat().make_one_shot_iterator().get_next()

def test_input_fn():
    test_dataset = tf.data.Dataset.from_tensor_slices((tf.constant(data_test), tf.constant(labels_test)))
    test_dataset = test_dataset.map(_parse_function)
    return test_dataset.shuffle(1000).batch(256).make_one_shot_iterator().get_next()

# %%
def cnn_model_fn(features, labels, mode):
	"""Model function for CNN."""
	# Input Layer
	input_layer = tf.reshape(features["image_data"], [-1, image_size, image_size, 3])
	tf.summary.image('image_data', features["image_data"])

	# Convolutional Layer #1
	conv1 = tf.layers.conv2d(
		inputs=input_layer,
		filters=32,
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu)

	# Pooling Layer #1
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

	# Convolutional Layer #1
	conv2 = tf.layers.conv2d(
		inputs=pool1,
		filters=64,
		kernel_size=[3, 3],
		padding="same",
		activation=tf.nn.relu)

	# Pooling Layer #1
	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

	# Convolutional Layer #2 and Pooling Layer #2
	conv3 = tf.layers.conv2d(
		inputs=pool2,
		filters=128,
		kernel_size=[5, 5],
		padding="same",
		activation=tf.nn.relu)
	pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

	# Dense Layer
	vol_size = int(image_size / 8)
	pool2_flat = tf.reshape(pool3, [-1, vol_size * vol_size * 128])
	dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
	dropout = tf.layers.dropout(inputs=dense, rate=0.2, training=(mode==tf.estimator.ModeKeys.TRAIN))
	dense1 = tf.layers.dense(inputs=dropout, units=1024, activation=tf.nn.relu)

	#dense1 = tf.layers.dense(inputs=dropout, units=1);
	#print(dense1)
	#dense1 = tf.reshape(dense1, [None, 3]);
	#print(dense1)
	#print(tf.layers.flatten(dropout))

	output_layer = tf.layers.dense(inputs=dense1, units=1, name="regression")
	# tf.summary.tensor_summary('output_layer', output_layer)

	#predictions = tf.tanh(regression, name="regression")
	predictions = tf.squeeze(output_layer, name="prediction")
	predictions = tf.add(predictions, tf.constant(year_offset), name="year")
	tf.summary.histogram('predictions', predictions)

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions={"year": predictions})

	tf.summary.histogram('labels', labels)

	# Calculate Loss (for both TRAIN and EVAL modes)
	average_loss = tf.losses.mean_squared_error(labels=labels, predictions=predictions)
	tf.summary.scalar('average_loss', average_loss)
	#loss = tf.reduce_mean(tf.squared_difference(predictions, labels))

	# Pre-made estimators use the total_loss instead of the average,
	# so report total_loss for compatibility.
	batch_size = tf.shape(labels)[0]
	total_loss = tf.to_float(batch_size) * average_loss
	tf.summary.scalar('total_loss', total_loss)

	# Configure the Training Op (for TRAIN mode)
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.AdamOptimizer(learning_rate=0.000001)
		train_op = optimizer.minimize(
		loss=average_loss,
		global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, train_op=train_op)

	# Add evaluation metrics (for EVAL mode)
	# Calculate root mean squared error
	rmse = tf.metrics.root_mean_squared_error(labels, predictions)
	mae = tf.metrics.mean_absolute_error(labels, predictions)

	# Add the rmse to the collection of evaluation metrics.
	eval_metrics = {
		"rmse": rmse,
		"mae": mae
	}

	return tf.estimator.EstimatorSpec(
		mode=mode,
		# Report sum of error for compatibility with pre-made estimators
		loss=total_loss,
		eval_metric_ops=eval_metrics)

tf.reset_default_graph()
regressor = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="C:/dev/poster_year/nn_year")


# %%
# Set up logging for predictions
tensors_to_log = {"year": "year"}
logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=100)

# Train the model
regressor.train(
    input_fn=train_input_fn,
    steps=800,
    hooks=[logging_hook])


# %%
eval_results = regressor.evaluate(input_fn=test_input_fn)
print(eval_results)

# %%
print(len(labels_test))
def predict_input_fn():
	start = 634
	to = 644
	print(labels_test[start:to])
	dataset = tf.data.Dataset.from_tensor_slices((tf.constant(data_test[start:to]), tf.constant(labels_test[start:to])))
	return dataset.map(_parse_function).batch(1).make_one_shot_iterator().get_next()

results = regressor.predict(input_fn=predict_input_fn, yield_single_examples=False)
print([p["year"] for p in list(results)])
# for result in list(results):
# 	print(result['year'])
