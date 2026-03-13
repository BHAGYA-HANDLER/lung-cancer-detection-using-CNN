# here we are importing all the necessary libraries for our project
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow import keras
from keras import layers
#defing the dataset path 
dataset_path = "IQ-OTHNCCD"
classes = [
    "Bengin cases",
    "Malignant cases",
    "Normal cases"
]

# now resizing all the images for cnn

IMG_SIZE = 128

X = []
Y = []

for label in classes:
    
    folder = os.path.join(dataset_path,label)
    
    for image in os.listdir(folder):
        
        img_path = os.path.join(folder,image)
        
        img = cv2.imread(img_path)
        
        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        
        X.append(img)
        Y.append(label)

X = np.array(X)
Y = np.array(Y)

# now we are encoding the labels using label encoder

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

Y = encoder.fit_transform(Y)

Y = tf.keras.utils.to_categorical(Y)

# now we are splitting the dataset into training and testing set

X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42
)

# now we build the cnn model

model = keras.Sequential([

layers.Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)),
layers.MaxPooling2D(2,2),

layers.Conv2D(64,(3,3),activation='relu'),
layers.MaxPooling2D(2,2),

layers.Conv2D(128,(3,3),activation='relu'),
layers.MaxPooling2D(2,2),

layers.Flatten(),

layers.Dense(128,activation='relu'),

layers.Dropout(0.3),

layers.Dense(3,activation='softmax')

])

model.summary()

# compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# training the model 

history = model.fit(

X_train,Y_train,
validation_data=(X_val,Y_val),
epochs=10,
batch_size=32

)

# at last we evsaluate the model on the validation set

from sklearn.metrics import classification_report

pred = model.predict(X_val)

Y_pred = np.argmax(pred,axis=1)
Y_true = np.argmax(Y_val,axis=1)

print(classification_report(Y_true,Y_pred))

model.save("lung_cancer_model.h5")