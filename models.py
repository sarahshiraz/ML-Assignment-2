import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import regularizers, models, layers, optimizers
from keras.datasets import cifar10



def model0():
    model0 = models.Sequential()
    model0.add(layers.Conv2D(32, (3, 3), padding='same',activation='relu', input_shape=(32, 32, 3)))
    model0.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model0.add(layers.Conv2D(64, (3, 3), padding='same',activation='relu'))
    model0.add(layers.Conv2D(64, (3, 3),activation='relu'))
    model0.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model0.add(layers.Conv2D(128, (2, 2), padding='same',activation='relu'))
    model0.add(layers.Conv2D(128, (2, 2),activation='relu'))
    model0.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model0.add(layers.Flatten())
    model0.add(layers.Dense(1024, activation='relu'))
    model0.add(layers.Dropout(0.25))
    model0.add(layers.Dense(10))
    model0.add(layers.Activation('softmax'))
    model0.compile(loss='categorical_crossentropy',  optimizer=optimizers.Adam(lr=1e-4), metrics=['acc'])
    return model0

def model1():
    model1 = models.Sequential()
    model1.add(layers.Conv2D(32, (3, 3), padding='same',activation='relu', input_shape=(32, 32, 3)))
    model1.add(layers.Conv2D(32, (3, 3),activation='relu'))
    model1.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model1.add(layers.Dropout(0.25))
    model1.add(layers.Conv2D(64, (3, 3), padding='same',activation='relu'))
    model1.add(layers.Conv2D(64, (3, 3),activation='relu'))
    model1.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model1.add(layers.Dropout(0.25))
    model1.add(layers.Conv2D(128, (2, 2), padding='same',activation='relu'))
    model1.add(layers.Conv2D(128, (2, 2),activation='relu'))
    model1.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model1.add(layers.Dropout(0.25))
    model1.add(layers.Flatten())
    model1.add(layers.Dense(1024, activation='relu'))
    model1.add(layers.Dropout(0.5))
    model1.add(layers.Dense(10))
    model1.add(layers.Activation('softmax'))
    model1.compile(loss='categorical_crossentropy',  optimizer=optimizers.Adam(lr=1e-4), metrics=['acc'])
    return model1

def model2():
  model2 = models.Sequential()
  model2.add(layers.Conv2D(32, (3, 3), activation='relu', padding = 'same', input_shape=(32, 32, 3)))
  model2.add(layers.MaxPooling2D(2, 2))
  model2.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model2.add(layers.MaxPooling2D(2, 2))
  model2.add(layers.Conv2D(128, (3, 3), activation='relu'))
  model2.add(layers.Conv2D(128, (3, 3), activation='relu'))
  model2.add(layers.MaxPooling2D(2, 2))
  model2.add(layers.Flatten())
  model2.add(layers.Dropout(0.5))
  model2.add(layers.Dense(1024, activation='relu'))
  model2.add(layers.Dropout(0.3))
  model2.add(layers.Dense(512, activation='relu'))
  model2.add(layers.Dense(10,activation ='softmax'))
  model2.compile(loss='categorical_crossentropy',  optimizer=optimizers.Adam(lr=1e-4), metrics=['acc'])
  return model2

num_classes = 10
weight_decay = 1e-4
def final_model():
    model = Sequential()
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=(32, 32, 3)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))


    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)


    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model
