from preprocessing import Preprocessing as myData
import tensorflow as tf


# import data

train_data, val_data, test_data = myData.importData()

train_data, val_data, test_data = myData.ProprocessData(train_data, val_data, test_data)


model = tf.keras.models.load_model('model_2.keras',compile=False)

validation_evaluation = model.evaluate(val_data, return_dict = True)

test_evaluation = model.evaluate(test_data, return_dict = True)

print(validation_evaluation)
print(test_evaluation)