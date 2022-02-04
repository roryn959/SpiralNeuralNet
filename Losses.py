import numpy as np

class Loss:
    def calculate(self, outputs, truths):
        sample_losses = self.forward(outputs, truths)
        data_loss = np.mean(sample_losses)

        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, preds, truths):
        #Number of samples in each batch
        n_samples = len(preds)
        #Data clipped to prevent division by 0. Clip both sides to ensure no drag to one side
        preds_clipped = np.clip(preds, 1e-7, 1 - 1e-7)
        #Probabilities for target values - only if categorical labels
        if len(truths.shape) == 1:
            corrected_confidences = preds_clipped[range(n_samples), truths]
        #Mask values - only for one-hot encoded labels
        elif len(truths.shape) == 2:
            corrected_confidences = np.sum(preds_clipped*truths, axis=1)
        #Losses. Log will be negative, time multiply by -1.
        losses = -np.log(corrected_confidences)
        return losses

    def backward(self, dvalues, truths):
        #Number of samples
        n_samples = len(dvalues)
        #Number of labels in samples. Use first sample to count
        n_labels = len(dvalues[0])
        #If labels are sparse, turn them into one-hot vector
        if len(truths.shape) == 1:
            truths = np.eye(n_labels)[truths]
        #Calculate gradient
        self.dinputs = -truths / dvalues
        #Normalise gradient
        self.dinputs /= samples
