from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28 * 28).astype("float32") / 255.
X_test = X_test.reshape(-1, 28 * 28).astype("float32") / 255.


class Dense(layers.Layer):
    def __init__(self, units):
        super(Dense, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            name='W',
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            name="b",
            shape=(self.units,),
            initializer="random_normal",
            trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class MyModel(keras.Model):

    def __init__(self, num_classes=10):
        super(MyModel, self).__init__()
        self.dense1 = Dense(64)
        self.dense2 = Dense(num_classes)
        self.relu = MyRelu()

    def call(self, input_tensor):
        X = self.relu(self.dense1(input_tensor))
        return self.dense2(X)


class MyRelu(layers.Layer):

    def __init__(self):
        super(MyRelu, self).__init__()

    def call(self, x):
        return tf.math.maximum(x, 0)


model = MyModel()
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"],
    )

model.fit(X_train, y_train, batch_size=32, epochs=2, verbose=2)
model.evaluate(X_test, y_test)