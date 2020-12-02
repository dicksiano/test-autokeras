import numpy as np
from config import Constants
import glob
from six.moves import cPickle

def p(k):
    epi = k.split(":")[0]
    t = k.split(":")[1]

    return epi + ":" + str(int(t)-1)

def read_data():
    x, y = [], []
    dictS, dictA = {}, {}

    for f in glob.glob('data/cpRnd/*.txt'):
        print(f)
        lines = open(f, 'r').readlines()

        for line in lines:
            vals = line.split()
            if len(vals) and vals[0] == '+++|||+++':
                dictS[f+"_"+vals[1]] = vals[2:]
            elif len(vals) and vals[0] == '***|||***':
                dictA[f+"_"+vals[1]] = vals[2:]

    for k in dictA.keys():
        previousK = p(k)
        if previousK in dictS:
            #print(k)
            x.append(dictS[previousK])
            y.append(dictA[k])

    x = np.array(x)
    y = np.array(y)
    
    assert( x.shape[0] == y.shape[0] )
    assert( x.shape[1] == Constants.INPUT_SIZE )
    assert( y.shape[1] == Constants.OUT_SIZE )

    target_path = 'meta_cpRnd'
    print('Saving: ',target_path)
    with open(target_path + '.pkl', 'wb') as cPickle_file:
        cPickle.dump([x, y], cPickle_file, protocol=cPickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    read_data()