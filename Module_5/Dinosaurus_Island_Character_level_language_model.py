import random
import pprint
import copy
import numpy as np

from utils import *
from numpy.random import rand


def clip(gradients, maxValue):
    '''
    Inp
    gradients: contains "dWaa", "dWax", "dWya", "db", "dby"
    maxValue
    
    Out:
        gradients: dictionary with the clipped gradients.
    '''
    gradients = copy.deepcopy(gradients)
    print(help(copy))


    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']
   
    ### START CODE HERE ###
    # Clip to mitigate exploding gradients, loop over [dWax, dWaa, dWya, db, dby]. (≈2 lines)
    for gradient in [dWax, dWaa, dWya, db, dby]:
        np.clip(gradient, -maxValue, maxValue, out = gradient)
    ### END CODE HERE ###
    
    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}
    
    return gradients


def Exercise_1() -> None:
	def clip_test(target, mValue):
	    print(f"\nGradients for mValue={mValue}")
	    np.random.seed(3)
	    
	    dWax = randn(5, 3) * 10
	    dWaa = randn(5, 5) * 10
	    dWya = randn(2, 5) * 10
	    
	    db = randn(5, 1) * 10
	    dby = randn(2, 1) * 10
	    
	    gradients = {"dWax": dWax, "dWaa": dWaa, "dWya": dWya, "db": db, "dby": dby}

	    gradients2 = target(gradients, mValue)
	    print("gradients[\"dWaa\"][1][2] =", gradients2["dWaa"][1][2])
	    print("gradients[\"dWax\"][3][1] =", gradients2["dWax"][3][1])
	    print("gradients[\"dWya\"][1][2] =", gradients2["dWya"][1][2])
	    print("gradients[\"db\"][4] =", gradients2["db"][4])
	    print("gradients[\"dby\"][1] =", gradients2["dby"][1])
	    
	    for grad in gradients2.keys():
	        valuei = gradients[grad]
	        valuef = gradients2[grad]
	        mink = np.min(valuef)
	        maxk = np.max(valuef)
	        assert mink >= -abs(mValue), f"Problem with {grad}. Set a_min to -mValue in the np.clip call"
	        assert maxk <= abs(mValue), f"Problem with {grad}.Set a_max to mValue in the np.clip call"
	        index_not_clipped = np.logical_and(valuei <= mValue, valuei >= -mValue)
	        assert np.all(valuei[index_not_clipped] == valuef[index_not_clipped]), f" Problem with {grad}. Some values that should not have changed, changed during the clipping process."
	    
	    print("\033[92mAll tests passed!\x1b[0m")

	clip_test(clip, 10)
	clip_test(clip, 5)



	return None    
###############################################################################


###############################################################################
def main() -> None:
	data = open('dinos.txt', 'r').read()
	
	data = sorted(data.lower())
	chars = sorted(list(set(data)))
	print(f'There are {len(data)} total characters and {len(chars)} unique characters in your data.')

	char_to_index = {chard:index for index, char in enumerate(chars)}
	index_to_char = {index: char for index, char in enumerate(chars)}

	Exercise_1()
	Exercise_2
	Exercise_3
	Exercise_4

	return None


if __name__ == '__main__':
	main()