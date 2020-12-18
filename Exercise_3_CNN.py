# this program introduces us into the world of Convolutional Neural Networks (CNN)

import tensorflow as tf

# Callback
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.998):
            print("Reached 99.8% accuracy so cancelling training!")
            self.model.stop_training=True

callbacks=myCallback()
minst=tf.keras.datasets.fashion_mnist

# loading the data
(training_images,training_labels), (test_images,test_labels)=minst.load_data()

# reshaping and normalizing the data
# in reshaping the 60000 examples are reshaped and the 1 in the end is to specify the byte
training_images=training_images.reshape(60000,28,28,1)
training_images=training_images/255.0
test_images=test_images.reshape(10000,28,28,1)
test_images=test_images/255.0

# creating the convolution and the layers
model=tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# compiling
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(training_images,training_labels,epochs=20,callbacks=[callbacks])
test_loss=model.evaluate(test_images,test_labels)