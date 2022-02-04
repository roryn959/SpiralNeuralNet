import numpy as np
from Losses import Loss_CategoricalCrossEntropy

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        self.inputs = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        #Remember input values
        self.inputs = inputs
        #Get unormalised probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        #Normalise probabilities for each sample
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def backward(self, dvalues):
        #Creat uninitialised array
        self.dinputs = np.empty_like(dvalues)
        #Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            #Flatten output array
            single_output = single_output.reshape(-1, 1)
            #Calculate Jacobian matrix of output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            #Calculate sample-wise gradient and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

class Activation_Softmax_Loss_CategoricalCrossEntropy():
    #Combined softmax and cross-entropy loss for faster backpropagation
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()

    def forward(self, inputs, truths):
        #Pass through activation function
        self.activation.forward(inputs)
        #Update output
        self.output = self.activation.output
        #Return the calculated loss value
        return self.loss.calculate(self.output, truths)

    def backward(self, dvalues, truths):
        #Number of samples
        n_samples = len(dvalues)
        #If labels are one-hot encoded, turn them into discrete values
        if len(truths.shape) == 2:
            truths = np.argmax(truths, axis=1)
        #Copy so we can safely modify
        self.dinputs = dvalues.copy()
        #Calculate gradient
        self.dinputs[range(n_samples), truths] -= 1
        #Normalise gradient
        self.dinputs = self.dinputs / n_samples







    
