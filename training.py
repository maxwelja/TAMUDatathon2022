import numpy as np
import tensorflow as tf
from glob import glob
from PIL import Image
from itertools import permutations
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import utils
from model import make_model
import os

def train():
    model = make_model()
    X_train,y_train=utils.load_all('train/',500)

    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1)

    model.fit(X_train, y_train,epochs=3,callbacks=[cp_callback],validation_split=0.03)

    model.save('savedmodel')
#print(np.argmax(model.predict(X_train),axis=1).tolist())