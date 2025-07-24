import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

print(f"Tensorflow: {tf.__version__}")
#print(f"Keras: {keras.__version__}")

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

print(X_train_full.shape)
print(X_train_full.dtype)

X_valid, X_train = X_train_full[:5000]/255.0, X_train_full[5000:]/255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
output_length = len(class_names)

print(class_names[y_train[0]])
print(X_train.shape[1:]) # [28,28]

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=X_train.shape[1:])) # X_train.reshape(-1, 1)
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dense(100, activation=keras.activations.relu))
model.add(keras.layers.Dense(output_length, activation='softmax'))

model2 = keras.models.Sequential([
    keras.layers.Flatten(input_shape=X_train.shape[1:]),
    keras.layers.Dense(300, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(output_length, activation='softmax')
])

print(model.summary())
print(model2.summary())

print(model.layers)
print(model.layers[0].name)
print(model.get_layer("dense").name)

hidden_layer = model.get_layer("dense")
weights, biases = hidden_layer.get_weights()
print(weights.shape)
print(weights)
print(biases.shape)
print(biases)

model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model2.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer=keras.optimizers.SGD(), metrics=[keras.metrics.sparse_categorical_accuracy])

history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))

'''
 Instead of passing a validation set using the validation_data
 argument, you could instead set validation_split to the ratio of
 the training set that you want Keras to use for validation (e.g., 0.1).
 
 class_weight argument
 sample_weight argument (it supersedes class_weight)
'''

print(history.params)
#print(history.epoch)
#print(history.history)

print(model.evaluate(X_test, y_test))


'''
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
'''

X_new = X_test[:3]
# for each instance the model estimates one probability per class
y_proba = model.predict(X_new)
print(y_proba.round(2))

# If you only care about the class with the highest estimated probability (even if that probability is quite low) then you can use the predict_classes() method instead
y_pred = model.predict_classes(X_new)
print(y_pred)

print(np.array(class_names)[y_pred])

print(y_test[:3])
