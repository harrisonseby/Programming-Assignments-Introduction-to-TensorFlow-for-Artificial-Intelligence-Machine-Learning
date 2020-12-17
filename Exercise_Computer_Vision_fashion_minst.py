import tensorflow as tf
import matplotlib.pyplot as plt

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.6):
            print("\nReached 60% accuracy Cancel training")
            self.model.stop_training=True
callbacks=myCallback()
mnist=tf.keras.datasets.fashion_mnist

# load data
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()

# displaying the data
plt.imshow(train_images[0])
print(train_labels[0])
print(train_images[0])

# normalizing the data
train_images=train_images/255.0
test_images=test_images/255.0

# creating the neural network
# first layer is flattened
# layer 2 is of 128 units and the activation is relu function
# layer 3 is the output layer with 10 units and activation function is softmax
model=tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                 tf.keras.layers.Dense(128,activation=tf.nn.relu),
                                 tf.keras.layers.Dense(10,activation=tf.nn.softmax)])


model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images,train_labels,epochs=10,callbacks=[callbacks])

model.evaluate(test_images,test_labels)

