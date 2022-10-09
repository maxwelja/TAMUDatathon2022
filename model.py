import tensorflow as tf
from tensorflow.keras.layers import TimeDistributed as td
from tensorflow.keras.layers import Conv2D, Flatten, Dense, ZeroPadding2D, Activation
from tensorflow.keras.layers import MaxPooling2D, Dropout, BatchNormalization, Reshape


def make_model():

    model = tf.keras.models.Sequential()

    model.add(td(ZeroPadding2D(2), input_shape=(4, 64, 64, 3)))

    model.add(td(Conv2D(50, kernel_size=(3,3), padding='same', activation='relu', strides=2)))
    model.add(td(BatchNormalization()))
    model.add(td(MaxPooling2D()))

    model.add(td(Conv2D(100, kernel_size=(3,3), padding='same', activation='relu', strides=2)))
    model.add(td(BatchNormalization()))
    #model.add(td(Dropout(0.5)))

    model.add(td(Conv2D(100, kernel_size=(3,3), padding='same', activation='relu', strides=2)))
    model.add(td(BatchNormalization()))
    #model.add(td(Dropout(0.5)))

    model.add(td(Conv2D(200, kernel_size=(3,3), padding='same', activation='relu', strides=1)))
    model.add(td(BatchNormalization()))
    #model.add(td(Dropout(0.5)))

    model.add(Flatten())

    model.add(Dense(600, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(400, activation='relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.5))
    model.add(Dense(16))
    model.add(Reshape((4, 4)))
    model.add(Activation('softmax'))

    adam = tf.keras.optimizers.Adam(learning_rate=.001)
    model.compile(metrics='accuracy',loss='sparse_categorical_crossentropy', optimizer=adam)

    model.get_config()

    return model