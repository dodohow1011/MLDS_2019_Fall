from keras.models import Sequential
from keras.layers import Dense, Activation

def two_layers():
    model = Sequential()
    model.add(Dense(input_dim=1, units=28, activation='relu'))
    model.add(Dense(units=27, activation='relu'))
    model.add(Dense(units=1, activation='linear'))
    model.summary()
    return model
