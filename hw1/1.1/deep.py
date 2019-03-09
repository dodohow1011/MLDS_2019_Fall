from keras.models import Sequential
from keras.layers import Dense

numUnits = 9

def Deep():
    model = Sequential()
    model.add(Dense(input_dim=1, units=numUnits, activation='relu'))
    model.add(Dense(input_dim=1, units=numUnits, activation='relu'))
    model.add(Dense(input_dim=1, units=numUnits, activation='relu'))
    model.add(Dense(input_dim=1, units=numUnits, activation='relu'))
    model.add(Dense(input_dim=1, units=1,  activation='linear'))
    model.summary()
    return model
