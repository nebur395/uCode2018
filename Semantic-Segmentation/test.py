import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import numpy as np
import random
import math
import os
import argparse
import copy
import time
from Loader import Loader
from imgaug import augmenters as iaa
import imgaug as ia
from augmenters import get_augmenter
import tensorflow.contrib.slim as slim
import Network
import cv2
random.seed(os.urandom(9))
#tensorboard --logdir=train:./logs/train,test:./logs/test/

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="Dataset to train", default='./clothes')  # 'Datasets/MNIST-Big/'
parser.add_argument("--dimensions", help="Temporal dimensions to get from each sample", default=3)
parser.add_argument("--tensorboard", help="Monitor with Tensorboard", default=0)
parser.add_argument("--augmentation", help="Image augmentation", default=1)
parser.add_argument("--init_lr", help="Initial learning rate", default=5e-3)
parser.add_argument("--min_lr", help="Initial learning rate", default=3e-7)
parser.add_argument("--init_batch_size", help="batch_size", default=1)
parser.add_argument("--max_batch_size", help="batch_size", default=1)
parser.add_argument("--n_classes", help="number of classes to classify", default=3)
parser.add_argument("--ignore_label", help="class to ignore", default=255)
parser.add_argument("--epochs", help="Number of epochs to train", default=2)
parser.add_argument("--width", help="width", default=224)
parser.add_argument("--height", help="height", default=224)
parser.add_argument("--save_model", help="save_model", default=1)
args = parser.parse_args()



# Hyperparameter
init_learning_rate = float(args.init_lr)
min_learning_rate = float(args.min_lr)
augmentation = bool(int(args.augmentation))
save_model = bool(int(args.save_model ))
tensorboard = bool(int(args.tensorboard))
init_batch_size = int(args.init_batch_size)
max_batch_size = int(args.max_batch_size)
total_epochs = int(args.epochs)
width = int(args.width)
n_classes = int(args.n_classes)
ignore_label = int(args.ignore_label)
height = int(args.height)
channels = int(args.dimensions)
change_lr_epoch = math.pow(min_learning_rate/init_learning_rate, 1.0/total_epochs)
change_batch_size = (max_batch_size - init_batch_size) / float(total_epochs - 1)

loader = Loader(dataFolderPath=args.dataset, n_classes=n_classes, problemType = 'segmentation', width=width, height=height, ignore_label = ignore_label)
testing_samples = len(loader.image_test_list)
training_samples = len(loader.image_train_list)


# For Batch_norm or dropout operations: training or testing
training_flag = tf.placeholder(tf.bool)

# Placeholder para las imagenes.
x = tf.placeholder(tf.float32, shape=[None, height, width, channels], name='input')
label = tf.placeholder(tf.float32, shape=[None, height, width, n_classes], name='output')
mask_label = tf.placeholder(tf.float32, shape=[None, height, width, n_classes], name='mask')
# Placeholders para las clases (vector de salida que seran valores de 0-1 por cada clase)

# Network
output = Network.complex(input_x=x, n_classes=n_classes, width=width, height=height, channels=channels, training=training_flag)
shape_output = output.get_shape()
label_shape = label.get_shape()

predictions = tf.reshape(output, [-1, shape_output[1]* shape_output[2] , shape_output[3]]) # tf.reshape(output, [-1])
labels = tf.reshape(label, [-1, label_shape[1]* label_shape[2] , label_shape[3]]) # tf.reshape(output, [-1])
output_image = tf.expand_dims(tf.cast(tf.argmax(output, 3), tf.float32), -1)
mask_labels = tf.reshape(mask_label, [-1, label_shape[1]* label_shape[2] , label_shape[3]]) # tf.reshape(output, [-1])


 
# Metrics

correct_prediction = tf.equal(tf.argmax(labels, 2), tf.argmax(predictions, 2))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
correct_prediction = tf.equal(tf.argmax(predictions, 2), tf.argmax(labels, 2))

correct_prediction_masked=tf.cast(correct_prediction, tf.float32)*tf.reduce_mean(mask_labels, axis=2)
sum_correc_masked=tf.reduce_sum(correct_prediction_masked)
sum_mask=tf.reduce_sum(tf.reduce_mean(mask_labels, axis=2))
accuracy = sum_correc_masked/sum_mask


acc, acc_op  = tf.metrics.accuracy(tf.argmax(labels, 2),tf.argmax(predictions, 2))
mean_acc, mean_acc_op = tf.metrics.mean_per_class_accuracy(tf.argmax(labels, 2), tf.argmax(predictions, 2), n_classes)
miou, miou_op = tf.metrics.mean_iou(tf.argmax(labels, 2), tf.argmax(predictions, 2), n_classes)

 
 # with tf.device('/cpu:0'):
import time


saver = tf.train.Saver(tf.global_variables())

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())
	ckpt = tf.train.get_checkpoint_state('./model_cloth')  # './model/best'
	ckpt_best = tf.train.get_checkpoint_state('./model_cloth/best')  # './model/best'
	if ckpt_best and tf.train.checkpoint_exists(ckpt_best.model_checkpoint_path):
		saver.restore(sess, ckpt_best.model_checkpoint_path)
		print('pesos cargados')

	img = cv2.imread('IMG_20180309_214412.jpg')
	img = np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))








	first = time.time()
	predictions = sess.run([output], feed_dict={x: img, training_flag : False})
	second = time.time()
	print(str(second - first) + " seconds to load")

	#output_image = np.reshape(output_image, (output_image.shape[1], output_image.shape[2], output_image.shape[3]))

	output_image=np.argmax(predictions[0], 3)
	output_image=np.reshape(output_image, (224,224,1))

	img=img[0,:,:,:]
	img2=copy.deepcopy(img[:,:,:])
	# 	img[output_image==8,:] = 0

	a=img[:,:,0]
	b=img[:,:,1]
	print(img.shape)

	print(a.shape)
	print(output_image.shape)

	c=img[:,:,2]
	kernel = np.ones((7,7),np.uint8)
	a=np.reshape(a, (224,224,1))
	b=np.reshape(b, (224,224,1))
	c=np.reshape(c, (224,224,1))
	a[output_image==1] = 0
	b[output_image==1] = 0
	c[output_image==1] = 0
	a = cv2.morphologyEx(a, cv2.MORPH_CLOSE, kernel)
	b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, kernel)
	c = cv2.morphologyEx(c, cv2.MORPH_CLOSE, kernel)
	print(a.shape)
	print(img.shape)
	img[:,:,0]=a[:,:]
	img[:,:,1]=b[:,:]
	img[:,:,2]=c[:,:]
	output_image[output_image==2] = 255





		

	print(np.unique(output_image))
	cv2.imshow('mask',output_image.astype(np.uint8))
	cv2.imshow('image',img)
	cv2.imshow('image real',img2)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
# mejor complex sin regularizer y droput: 0.80/0.77, 0.44, 0.66

