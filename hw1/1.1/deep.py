from keras.models import Sequential
from keras.layers import Dense

def Deep():
    model = Sequential()
    model.add(Dense(input_dim=1, units=16, activation='relu'))
    model.add(Dense(input_dim=1, units=16, activation='relu'))
    model.add(Dense(input_dim=1, units=16, activation='relu'))
    model.add(Dense(input_dim=1, units=16, activation='relu'))
    model.add(Dense(input_dim=1, units=1,  activation='linear'))
    model.summary()
    return model
