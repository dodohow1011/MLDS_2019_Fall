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
    from shallow import Shallow
    from two_layers import two_layers
    # model = Deep()
    # model = Shallow()
    model = two_layers()
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])

    history = model.fit(train_x, train_y, batch_size=64, epochs=2000)

    import pickle
    with open('two_layers_history.pickle', 'wb') as f:
    # with open('shallow_history.pickle', 'wb') as f:
    # with open('deep_history.pickle', 'wb') as f:
        pickle.dump(history.history, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
