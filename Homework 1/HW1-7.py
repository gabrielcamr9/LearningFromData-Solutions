import os
import sys
import numpy as np
from tools import ProgressIterator, random_target_function, random_set, pla, weight_error, output, experiment

above_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, above_dir)

np.random.seed(1346)

# THE PERCEPTRON LEARNING ALGORITHM


def trial(in_sample, out_sample):
    target_function = random_target_function()
    training_set = random_set(in_sample, target_function)
    initial_weight = np.zeros(len(training_set.x[0]))
    weight, iterations = pla(training_set.z, training_set.y, initial_weight, True)
    testing_set = random_set(out_sample, target_function)
    out_error = weight_error(weight, testing_set.z, testing_set.y)
    return out_error, iterations


def myTrial(in_sample, out_sample):
    target_function = random_target_function()
    #w0 es agregado por default
    training_set = random_set(in_sample, target_function)
    initial_weight = np.zeros(len(training_set.x[0]))
    weight, iterations = myOwnPlaImplementation(training_set.z, training_set.y, initial_weight)
    testing_set = random_set(out_sample, target_function)
    out_error = weight_error(weight, testing_set.z, testing_set.y)
    return out_error, iterations

#Esta es la función que hice yo
def myOwnPlaImplementation(input,output,weights):
    x_t = input.transpose()
    w_t = weights.transpose()
    converged = False
    iterations = 0
    while True:
        iterations += 1
        h = np.sign(np.dot(w_t,x_t))
        for i in range (len(h)):
            if(h[i] != output[i]):
                w_t = w_t + output[i]*input[i]
                break
            if(i==len(h)-1):
                converged=True
        if(converged):
            break
    return w_t, iterations


def main():
    output(simulations)


def simulations():
    que = {}
    progress_iterator = ProgressIterator(2)

    progress_iterator.next()
    out_error, iterations = experiment(myTrial, [10, 100], 1000)
    que[7] = ("iterations :", iterations)
    que[8] = ("out of sample error :", out_error)

    progress_iterator.next()
    out_error, iterations = experiment(myTrial, [100, 100], 1000)
    que[9] = ("iterations :", iterations)
    que[10] = ("out of sample error :", out_error)
    return que


if __name__ == "__main__":
    main()
