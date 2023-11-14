# library import
# import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import pyttsx3
# import cv2yy


engine = pyttsx3.init()
engine.setProperty('voice', 'f2')
engine.say('Initializing Note Prediction')
engine.runAndWait()
# data preprocessing

#training image processing
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

training_set = train_datagen.flow_from_directory(
    'training_set', target_size=(64,64), batch_size=32, class_mode='categorical')


#test image processing

test_datagen = ImageDataGenerator(
        rescale=1./255)

test_set = test_datagen.flow_from_directory(
    'test_set', target_size=(64,64), batch_size=32, class_mode='categorical')


#building model

cnn = tf.keras.models.Sequential()

#building convolution layer

cnn.add(tf.keras.layers.Conv2D(filters=64 , kernel_size=3 , activation='relu' , input_shape=[64,64,3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=64 , kernel_size=3 , activation='relu' ))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2 , strides=2))



cnn.add(tf.keras.layers.Dropout(0.5))

cnn.add(tf.keras.layers.Flatten())


cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=7 , activation='softmax'))


cnn.compile(optimizer = 'rmsprop' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])


cnn.fit(x = training_set , validation_data = test_set , epochs = 37)

# 
# input

from keras.preprocessing import image
test_image = image.load_img('prediction/twh.jpg',target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = cnn.predict(test_image)
training_set.class_indices




if result[0][0]==1:
    engine.say("Fiftynote")
    engine.runAndWait()
    print('Fiftynote')
elif result[0][1]==1:
    engine.say("FiveHundrednote")
    engine.runAndWait()
    print('FiveHundrednote')
elif result[0][2]==1:
    engine.say('OneHundrednote')
    engine.runAndWait()
    print('OneHundrednote')
elif result[0][3]==1:
    engine.say('Tennote')
    engine.runAndWait()
    print('Tennote')
elif result[0][4]==1:
    engine.say('Twentynote')
    engine.runAndWait()
    print('Twentynote')
elif result[0][5]==1:
    engine.say('TwoHundrednote')
    engine.runAndWait()
    print("TwoHundrednote")
elif result[0][6]==1:
    engine.say('TwoThousandnote')
    engine.runAndWait()
    print('TwoThousandnote')


print(result)



tf.keras.models.save_model(cnn,'my_model.hdf5')



