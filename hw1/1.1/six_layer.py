from keras.layers import Conv2D, Dense
from keras.models import Sequential
from keras.layers import MaxPooling2D, Flatten

def Six_layers():
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
    model.add(Conv2D(7, kernel_size=(3,3), activation='relu'))
    model.add(Conv2D(7, kernel_size=(3,3), activation='relu'))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(7, kernel_size=(3,3), activation='relu'))

    model.add(Conv2D(7, kernel_size=(3,3), activation='relu'))
    model.add(Conv2D(7, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(units=10, activation='softmax'))
    model.summary()
    return model
