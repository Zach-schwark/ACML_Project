import tensorflow as tf


#import  data
class Preprocessing():
    
    
    data_augmentation_layers = [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ]

    def  data_augmentation(self,images):
        for layer in  self.data_augmentation_layers:
            images = layer(images)
        return images
    def preprocess_image(self,image, label):
        image = tf.cast(image, tf.float32) / 255.0  # Normalize pixel values to [0, 1]
        image = self.data_augmentation(image)
        return image, label
    

    def importData():
    
        IMG_SIZE = 224
        BATCH_SIZE = 32

        train_data = tf.keras.utils.image_dataset_from_directory(
            "data/train",
            shuffle = True,
            image_size = (IMG_SIZE,IMG_SIZE),
            batch_size = BATCH_SIZE
        )

        test_data = tf.keras.utils.image_dataset_from_directory(
            directory = "data/test",
            shuffle = True,
            image_size = (IMG_SIZE, IMG_SIZE),
            batch_size = BATCH_SIZE
        )

        val_data = tf.keras.utils.image_dataset_from_directory(
            directory = "data/valid",
            shuffle = True,
            image_size = (IMG_SIZE,IMG_SIZE),
            batch_size = BATCH_SIZE
        )
        
        return train_data, test_data, val_data


    def ProprocessData(self,train_data,val_data,test_data ):

        train_data = train_data.map(
            self.preprocess_image,
            num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)

        val_data = val_data.map(
            lambda img, label: (tf.cast(img, tf.float32) / 255.0, label),
            num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)

        test_data = test_data.map(
            lambda img, label: (tf.cast(img, tf.float32) / 255.0, label),
            num_parallel_calls=tf.data.AUTOTUNE
        ).prefetch(tf.data.AUTOTUNE)
        
        return train_data, val_data, test_data