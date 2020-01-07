from keras import applications
from keras.models import Model
from keras.layers import Input, Reshape, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from classification_models.keras import Classifiers

ResNet18, preprocess_input = Classifiers.get('resnet18')

scene_map = Input(shape=(8, 64, 64, 3))
img_height,img_width = 64,64
#base_model = applications.resnet34.ResNet34(weights= None, include_top=False, input_shape= (img_height,img_width,2))
base_model = ResNet18(input_shape=(64,64,3), weights='imagenet', include_top=False)
x = TimeDistributed(base_model)(scene_map)
x = TimeDistributed(Flatten())(x)
x = LSTM(128)(x)



model = Model(inputs = scene_map, outputs = x)
model.summary()