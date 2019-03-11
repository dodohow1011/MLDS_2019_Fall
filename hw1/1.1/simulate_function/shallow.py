from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Activation

def Shallow():
    model = Sequential()
    model.add(Dense(input_dim=1, units=100, activation='relu'))
    model.add(Dense(units=1, activation='linear'))
    model.summary()
    return model
