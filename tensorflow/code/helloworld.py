import tensorflow.keras as tf

mnist = tf.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.models.Sequential([
  tf.layers.Flatten(input_shape=(28, 28)),
  tf.layers.Dense(128, activation='relu'),
  tf.layers.Dropout(0.2),
  tf.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)