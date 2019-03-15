import math
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import sys
PI = math.pi

def main():
    ##########################
    ## sin(3*pi*x)/(3*pi*x) ##
    ##########################
    train_x = np.arange(0,1,0.005).reshape((-1,1))
    train_y = np.array([math.sin(3*PI*x)/(3*PI*x) for x in train_x]).reshape((-1,1))
    train_y[0] = 1

    from deep import Deep 
    from shallow import Shallow
    from two_layers import two_layers
    if len(sys.argv) != 2:
        print ('usage: python3.5 simulate.py deep/shallow/two_layers')
        sys.exit()
    if sys.argv[1] == 'deep':
        model = Deep()
    elif sys.argv[1] == 'shallow':
        model = Shallow()
    elif sys.argv[1] == 'two_layers':
        model = two_layers()
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])

    history = model.fit(train_x, train_y, batch_size=16, epochs=1500)
    
    if sys.argv[1] == 'deep':
        model.save('deep_model.h5')
    elif sys.argv[1] == 'shallow':
        model.save('shallow_model.h5')
    elif sys.argv[1] == 'two_layers':
        model.save('two_layers_model.h5')
    import pickle
    if sys.argv[1] == 'deep':
        with open('deep_history.pickle', 'wb') as f:
            pickle.dump(history.history, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif sys.argv[1] == 'shallow':
        with open('shallow_history.pickle', 'wb') as f:
            pickle.dump(history.history, f, protocol=pickle.HIGHEST_PROTOCOL)
    elif sys.argv[1] == 'two_layers':
        with open('two_layers_history.pickle', 'wb') as f:
            pickle.dump(history.history, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
