import numpy as np
import tensorflow as tf
from glob import glob
from PIL import Image
from itertools import permutations
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import utils
from model import get_model
import os

X_train,y_train=utils.load_all('train/',100)

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
model = get_model()
adam = tf.keras.optimizers.Adam(lr=.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=adam)
model.load_weights(checkpoint_path)
model.save('model')
#print(np.argmax(model.predict(X_train),axis=1).tolist())
#print("Restored model, accuracy: {:5.2f}%".format(100 * acc))