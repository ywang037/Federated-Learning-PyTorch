# On reproducing the results in classic FL paper

### Off-the-shelf code
For the vanilla federated learning paper *Communication-Efficient Learning of Deep Networks from Decentralized Data*, there is only few open-source code implementation:
1. The [repository where this project is forked from](https://github.com/AshwinRJ/Federated-Learning-PyTorch).
2. [This post](https://hackmd.io/@stefanhofman/rJCJCViOL), and there code can be [viewd on google colab](https://colab.research.google.com/drive/1sWdbt_a3Dya9TQKTB2k5p-kRJWiznGsb#scrollTo=nlFXcKdIBYBA).

Both of the above only implemented the experiment on MNIST and CIFAR10, no LSTM language model and related result is reproduced.

### Exact CNN model used in the vanilla paper

In the vanilla FL paper, they used two simple CNNs for experimenting with MNIST and CIFAR10. Below is the description of the CNN model for MNIST:
> A CNN with two 5x5 convolution layers (the first with 32 channels, the second with 64, each followed with 2x2 max pooling), a fully connected layer with 512 units and ReLu activation, and a final softmax output layer (1,663,370 total parameters).

Below is the description of the CNN model for CIFAR10:
> The model architecture was taken from the TensorFlow tutorial, which consists of two convolutional layers followed by two fully connected layers and then a linear transformation layer to produce logits, for a total of about 106 parameters.

The citation they refer to cannot be found as of now. However, TensorFlow tutorial on implementing a simple CNN can still be found [here](https://www.tensorflow.org/tutorials/images/cnn?hl=zh-cn). Alternatively, one can also refer to this webpage for the information and architecture of the example CNN.  
