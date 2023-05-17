from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np

import argparse
from datetime import datetime
import time

"""
Bi-directional Recurrent Neural Network Example.
Build a bi-directional recurrent neural network (LSTM) with TensorFlow 2.0.

REFs: https://github.com/aymericdamien/TensorFlow-Examples
"""

parser = argparse.ArgumentParser()
# LMS parameters
#lms_group = parser.add_mutually_exclusive_group(required=False)
parser.add_argument('--lms', dest='lms', action='store_true', help='Enable LMS')
parser.add_argument('--no-lms', dest='lms', action='store_false', help='Disable LMS (Default)')
parser.add_argument('batch_size', type=int,  help="batch size, e.g., 256")
parser.add_argument('height_width', type=int,  help="dataset scale, e.g., 32")
parser.add_argument('steps', type=int,  help="training steps, e.g., 10")
parser.set_defaults(lms=False)
args = parser.parse_args()

if args.lms:
    tf.config.experimental.set_lms_enabled(True)
    tf.experimental.get_peak_bytes_active(0)

# MNIST dataset parameters.
# total classes (0-9 digits).
num_classes = 100
# data features (img shape: 28*28).
img_h, img_w = args.height_width, args.height_width
num_features = img_h*img_w

# Training Parameters
learning_rate = 0.001
training_steps = args.steps
batch_size = args.batch_size
display_step = 1

# Network Parameters
# MNIST image shape is 28*28px, we will then handle 28 sequences of 28 timesteps for every sample.
num_input = img_h # number of sequences.
timesteps = img_w # timesteps.
#num_units = 32 # number of neurons for the LSTM layer.
#num_units = 3072 # number of neurons for the LSTM layer.
num_units = 5120 # number of neurons for the LSTM layer.

"""
# Prepare MNIST data.
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
"""

num_imgs = 5000
x_train = np.random.randn(num_imgs, img_h, img_w).astype(np.float32)
y_train = np.random.randn(num_imgs).astype(np.float32)

# x_train shape: (60000, 28, 28)
# y_train shape: (60000,)
print('x_train shape: {}'.format(x_train.shape))
print('y_train shape: {}'.format(y_train.shape))

# Convert to float32.
x_train = np.array(x_train, np.float32)
# Flatten images to 1-D vector of 784 features (28*28).
x_train = x_train.reshape([-1, img_h, img_w])
# Normalize images value from [0, 255] to [0, 1].
x_train = x_train / 255.

# Use tf.data API to shuffle and batch data.
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
shuffle_size = int(num_imgs/10)
train_data = train_data.repeat().shuffle(shuffle_size).batch(batch_size).prefetch(batch_size)

# Create LSTM Model.
class BiRNN(Model):
    # Set layers.
    def __init__(self):
        super(BiRNN, self).__init__()
        # Define 2 LSTM layers for forward and backward sequences.
        lstm_fw = layers.LSTM(units=num_units)
        lstm_bw = layers.LSTM(units=num_units, go_backwards=True)
        # BiRNN layer.
        self.bi_lstm = layers.Bidirectional(lstm_fw, backward_layer=lstm_bw)
        # Output layer (num_classes).
        self.out = layers.Dense(num_classes)

    # Set forward pass.
    def call(self, x, is_training=False):
        x = self.bi_lstm(x)
        x = self.out(x)
        if not is_training:
            # tf cross entropy expect logits without softmax, so only
            # apply softmax when not training.
            x = tf.nn.softmax(x)
        return x

# Build LSTM model.
birnn_net = BiRNN()

# Cross-Entropy Loss.
# Note that this will apply 'softmax' to the logits.
def cross_entropy_loss(x, y):
    # Convert labels to int 64 for tf cross-entropy function.
    y = tf.cast(y, tf.int64)
    # Apply softmax to logits and compute cross-entropy.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
    # Average loss across the batch.
    return tf.reduce_mean(loss)

# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)

# Adam optimizer.
optimizer = tf.optimizers.Adam(learning_rate)

# Optimization process.
def run_optimization(x, y):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        # Forward pass.
        pred = birnn_net(x, is_training=True)
        # Compute loss.
        loss = cross_entropy_loss(pred, y)

    # Variables to update, i.e. trainable variables.
    trainable_variables = birnn_net.trainable_variables

    # Compute gradients.
    gradients = g.gradient(loss, trainable_variables)

    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, trainable_variables))

# miliseconds
print(datetime.now().timetz())
time_list = []
# time in ms
cur_time = int(round(time.time()*1000))
# Run training for the given number of steps.
for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    # Run the optimization to update W and b values.
    run_optimization(batch_x, batch_y)
    next_time = int(round(time.time()*1000))
    time_list.append(next_time - cur_time)
    cur_time = next_time
    if step % display_step == 0:
        pred = birnn_net(batch_x, is_training=True)
        loss = cross_entropy_loss(pred, batch_y)
        acc = accuracy(pred, batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))
        print('peak active bytes(MB): {}'.format(tf.experimental.get_peak_bytes_active(0)/1024.0/1024.0))
        print('bytes in use(MB): {}'.format(tf.experimental.get_bytes_in_use(0)/1024.0/1024.0))

print('throughput: {} ms!!!'.format(np.average(np.array(time_list))))