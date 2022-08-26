import keras
import numpy as np
from keras_preprocessing.image import ImageDataGenerator

traning_dir = 'Learn data'
traning_datagen = ImageDataGenerator(rescale=1./255)

traning_generator = traning_datagen.flow_from_directory(traning_dir, target_size=(150, 150), class_mode='categorical')
