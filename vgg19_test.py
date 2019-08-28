from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model
import numpy as np
import datetime

base_model = VGG19(weights='imagenet')

img_path = 'inputs/fox_squirrel.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

before = datetime.datetime.now()

output = decode_predictions(base_model.predict(x))

print(output)
print("Took %s seconds" % (datetime.datetime.now() - before))
