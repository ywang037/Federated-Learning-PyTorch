# On reproducing the results in classic FL paper

### Off-the-shelf code
For the vanilla federated learning paper *Communication-Efficient Learning of Deep Networks from Decentralized Data*, there is only few open-source code implementation:
1. The [repository where this project is forked from](https://github.com/AshwinRJ/Federated-Learning-PyTorch).
2. [This post](https://hackmd.io/@stefanhofman/rJCJCViOL), and there code can be [viewd on google colab](https://colab.research.google.com/drive/1sWdbt_a3Dya9TQKTB2k5p-kRJWiznGsb#scrollTo=nlFXcKdIBYBA).

Both of the above only implemented the experiment on MNIST and CIFAR10, no LSTM language model and related result is reproduced.

Other code implementation can be found on [this page](https://paperswithcode.com/paper/communication-efficient-learning-of-deep) of the *Paper with Code*.

### Exact CNN model used in the vanilla paper

In the vanilla FL paper, they used two simple CNNs for experimenting with MNIST and CIFAR10. Below is the description of the CNN model for MNIST:
> A CNN with two 5x5 convolution layers (the first with 32 channels, the second with 64, each followed with 2x2 max pooling), a fully connected layer with 512 units and ReLu activation, and a final softmax output layer (1,663,370 total parameters).

Below is the description of the CNN model for CIFAR10:
> The model architecture was taken from the TensorFlow tutorial, which consists of two convolutional layers followed by two fully connected layers and then a linear transformation layer to produce logits, for a total of about 106 parameters.

The citation to which they referred in the paper cannot be found as of now. However, official TensorFlow tutorial on implementing a simple CNN can still be found [here](https://www.tensorflow.org/tutorials/images/cnn?hl=zh-cn). Alternatively, one can also refer to [this webpage](https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-tensorflow/) for the information and architecture of the example CNN.  

### Implementing the training

One need to take care of the training hyper-parameters that had been used in the vannila paper, e.g., batch size, epochs, learning rate and related decaying schemes:
1. To reproduce the FL training results, follow careflly what are described in the vanilla paper.
2. To reproduce the baseline training results, follow the paper and also check out how training goes like in other literatures that use MNIST and CIFAR10 (e.g., the paper *Closing the generalization gap of adaptive gradient methods in training deep neural networks*).
3. Read necessary literatures and see how to select these hyper parameters (e.g., paper *Optimal mini-batch size selection for fast gradient descent*, etc.)

### Work logs
#### 27 April 2021
For baseline training on CIFAR, it is found that both the CNN model created by WY and the one in AshwinRJ's repository cannot produce good test accuracy (both leads to final accuracies below 60%) after 100 epochs when a learning rate of 0.1, vanilla SGD, and a batch size of 100 is applied.

For both models, the training loss keeps reduced and then converges around 50 epochs, while the test accuracies first increase (to approximately 63%) then start to decline gradually below 60% (around 58%) only after about 15 epochs.

It looks like the models **might be overfitted**.

#### 28 April 2021
A fixed learning rate 0.01 does not show test accuracy drops as was observed in the previous experiment using lr=0.1, for same epoch number 100, batch size 100 and vanilla SGD. 

After 100 epochs, the training loss keeps declining and test accuracy converges around 60 epochs to an accuracy about 64%.