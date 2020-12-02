import numpy as np
from config import Constants
import glob
from six.moves import cPickle

def read_data():
    x, y = [], []

    for f in glob.glob('./data/cp0/*rank[1-2][0-9]__infer.txt'):
        print(f)
        lines = open(f, 'r').readlines()
        if lines[1].split()[0][0:4] == "2020-":
            lines = lines[1:]
        lines = lines[1:] # Remove datetime
        np.random.shuffle(lines) # Shuffling data

        for line in lines:
            vals = line.split()
            x.append( [ float(x) for x in vals[:Constants.INPUT_SIZE] ] )
            y.append( [ float(x) for x in vals[-Constants.OUT_SIZE:] ] )

    x = np.array(x)
    y = np.array(y)


    assert( x.shape[0] == y.shape[0] )
    assert( x.shape[1] == Constants.INPUT_SIZE )
    assert( y.shape[1] == Constants.OUT_SIZE )

    target_path = 'meta_pushrecov_cp0___test'
    print('Saving: ',target_path)
    with open(target_path + '.pkl', 'wb') as cPickle_file:
        cPickle.dump([x, y], cPickle_file, protocol=cPickle.HIGHEST_PROTOCOL)