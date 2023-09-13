import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
import joblib


def train_model():
    gen_train = ImageDataGenerator(rescale = 1/255, shear_range = 0.2, zoom_range = 0.2, brightness_range = (0.1, 0.5), horizontal_flip=True)

    TRAIN_DIRECTORY="TRAIN_IMAGES"

    train_data = gen_train.flow_from_directory(TRAIN_DIRECTORY, target_size = (224, 224), class_mode="categorical")

    # here i'm going to take input shape, weights and bias from imagenet and include top False means
    # i want to add input, flatten and output layer by my self

    vgg16 = VGG16(input_shape = (224, 224, 3), weights = "imagenet", include_top = False)

    for layer in vgg16.layers:
        layer.trainable = False
        
    x = layers.Flatten()(vgg16.output)

    # now let's add output layers or prediction layer

    prediction = layers.Dense(units = 2, activation="softmax")(x)

    # creating a model object

    model = tf.keras.models.Model(inputs = vgg16.input, outputs=prediction)
    model.summary()

    # now let's compile the model

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics =["accuracy"])

    result = model.fit(train_data, epochs = 28, steps_per_epoch=len(train_data))
    
    # Saving the Model
    joblib.dump(model, 'model.sav')



def waste_prediction(new_image):
    output_class = ["Bio-degradable", "Non Bio-degradable"]
    
    # Load Model
    model = joblib.load('model.sav')
    test_image = image.load_img(new_image, target_size = (224,224))
    plt.axis("on")
    plt.imshow(test_image)
    plt.show()

    test_image = image.img_to_array(test_image) / 255
    test_image = np.expand_dims(test_image, axis=0)

    predicted_array = model.predict(test_image)
    predicted_value = output_class[np.argmax(predicted_array)]
    predicted_accuracy = round(np.max(predicted_array) * 100, 2)

    print("Your waste material is ", predicted_value, " with ", predicted_accuracy, " % accuracy")
    return ('B' if predicted_value == 'Bio-degradable' else 'NB', predicted_accuracy)


# Train
# train_model()
# print(waste_prediction('TEST_IMAGES/bttl.jpg'))