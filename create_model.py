import tensorflow as tf
from tensorflow import keras
from keras import layers

model = keras.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Conv2D(8, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy')
model.save('models/lung_cancer_model.h5')
print('Saved dummy model to models/lung_cancer_model.h5')
