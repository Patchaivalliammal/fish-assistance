import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


DATA_DIR = 'data/images'
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 8 # increase later if you have time/GPU
MODEL_PATH = 'models/fish_classifier.h5'
LABELS_PATH = 'labels.txt'


os.makedirs('models', exist_ok=True)


# 1) Load datasets with an 80/20 train/val split
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
DATA_DIR,
validation_split=0.2,
subset='training',
seed=42,
image_size=IMG_SIZE,
batch_size=BATCH_SIZE
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
DATA_DIR,
validation_split=0.2,
subset='validation',
seed=42,
image_size=IMG_SIZE,
batch_size=BATCH_SIZE
)


class_names = train_ds.class_names
print('Classes:', class_names)


# 2) Cache + prefetch for speed
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# 3) Data augmentation
augmentation = keras.Sequential([
layers.RandomFlip('horizontal'),
layers.RandomRotation(0.05),
layers.RandomZoom(0.1),
])


# 4) Base model: MobileNetV2
base = tf.keras.applications.MobileNetV2(
input_shape=IMG_SIZE + (3,),
include_top=False,
weights='imagenet'
)
base.trainable = False # freeze for initial training
print('Saved labels to', LABELS_PATH)