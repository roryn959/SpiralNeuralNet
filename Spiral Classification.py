import matplotlib.pyplot as plt
import numpy as np
from nnfs.datasets import spiral_data, vertical_data
from Layers import *
from Activations import *
from Losses import *
from Optimisers import *

x, y = spiral_data(samples=100, classes=3)
dense1 = Layer_Dense(2, 64)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(64, 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()

#optimiser = Optimiser_SGD(learning_rate=1, decay=1e-3, momentum=0.9)
#optimiser = Optimiser_AdaGrad(learning_rate=1, decay=1e-4)
#optimiser = Optimiser_RMSProp(learning_rate=0.02, decay=1e-5, rho=0.999)
optimiser = Optimiser_Adam(learning_rate=0.05, decay=1e-6)

for epoch in range(10001):
    dense1.forward(x)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)
    predictions = np.argmax(loss_activation.output, axis=1)
    
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==y)

    if not epoch % 100:
        print(f"Epoch: {epoch}, Accuracy: {accuracy:.3f}, Loss: {loss:.3f}, LR: {optimiser.current_learning_rate}")

    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    optimiser.pre_update_params()
    optimiser.update_params(dense1)
    optimiser.update_params(dense2)
    optimiser.post_update_params()




        
