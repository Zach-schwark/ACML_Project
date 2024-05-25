import tensorflow as tf
from tensorflow import keras as keras
import matplotlib.pyplot as plt
import keras_tuner as kt
from preprocessing import Preprocessing as myData

IMG_SIZE = 224
BATCH_SIZE = 32
CHANNELS =3
input_shape = (IMG_SIZE, IMG_SIZE, CHANNELS)


def Model(input_shape):

    model = keras.Sequential([
        keras.layers.Conv2D(32,kernel_size=3,input_shape=input_shape),
        keras.layers.MaxPool2D((2,2)),
        
        keras.layers.Conv2D(64,kernel_size=3),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPool2D((2,2)),
        
        keras.layers.Conv2D(64,kernel_size=3),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPool2D((2,2)),
        
        keras.layers.Conv2D(64,kernel_size=3),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPool2D((2,2)),
        
        keras.layers.Conv2D(64,kernel_size=3),
        keras.layers.BatchNormalization(),
        keras.layers.ReLU(),
        keras.layers.MaxPool2D((2,2)),
        
        keras.layers.Flatten(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(256, activation='relu', activity_regularizer = tf.keras.regularizers.L2()),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(10, activation='softmax'),
    ])

    return model

# import data

train_data, val_data, test_data = myData.importData()

train_data, val_data, test_data = myData.ProprocessData(train_data, val_data, test_data)




model = Model(input_shape)
model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.002),
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits =False),metrics=['accuracy'])

earlystopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=20,restore_best_weights=True)

#,callbacks=[earlystopping]

history = model.fit(
train_data,
epochs = 50,
batch_size = 32,
validation_data = val_data,callbacks=[earlystopping])


model.save(filepath = "model_2.keras")