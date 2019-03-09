import math
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

PI = math.pi

def main():
    ##########################
    ## sin(3*pi*x)/(3*pi*x) ##
    ##########################
    train_x = np.arange(0,1,0.001).reshape((-1,1))
    train_y = np.array([math.sin(3*PI*x)/(3*PI*x) for x in train_x]).reshape((-1,1))
    train_y[0] = 1

    from deep import Deep 

    model = Deep()
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    history = model.fit(train_x, train_y, batch_size=32, epochs=200)
    print(history)

if __name__ == '__main__':
    main()
