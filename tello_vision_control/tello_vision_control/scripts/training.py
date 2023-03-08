""" Script to train the hand pose classifier 
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tello_vision_control.utils import classification_utils

dataset = 'model2/keypoint2.csv'
model_save_path = 'model2/keypoint_classifier2.hdf5'
tflite_save_path = 'model2/keypoint_classifier2.tflite'
NUM_CLASSES = 3

### Prepare data
X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))
X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75, random_state=42)

### Create classifier
classifier = classification_utils.create_landmarks_classifier(NUM_CLASSES)
classifier.summary()


### Prepare training
cp_callback = tf.keras.callbacks.ModelCheckpoint(model_save_path, verbose=1, save_weights_only=False)
es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)

classifier.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


### Train
classifier.fit(
    X_train,
    y_train,
    epochs=1000,
    batch_size=128,
    validation_data=(X_test, y_test),
    callbacks=[cp_callback, es_callback]
)

val_loss, val_acc = classifier.evaluate(X_test, y_test, batch_size=128)

### Save
# classifier = tf.keras.models.load_model(model_save_path)
classifier.save(model_save_path, include_optimizer=False)
classification_utils.save_classifier_to_TFLite(classifier, tflite_save_path)