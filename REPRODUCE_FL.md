# On reproducing the results in classic FL paper

### I. Off-the-shelf code
For the vanilla federated learning paper *Communication-Efficient Learning of Deep Networks from Decentralized Data*, there is only few open-source code implementation:
1. The [repository where this project is forked from](https://github.com/AshwinRJ/Federated-Learning-PyTorch).
2. [This post](https://hackmd.io/@stefanhofman/rJCJCViOL), and there code can be [viewd on google colab](https://colab.research.google.com/drive/1sWdbt_a3Dya9TQKTB2k5p-kRJWiznGsb#scrollTo=nlFXcKdIBYBA).

Both of the above only implemented the experiment on MNIST and CIFAR10, no LSTM language model and related result is reproduced.

Other code implementation can be found on [this page](https://paperswithcode.com/paper/communication-efficient-learning-of-deep) of the *Paper with Code*.

### II. Exact CNN models used in the vanilla paper

In the vanilla FL paper, they used two simple CNNs for experimenting with MNIST and CIFAR10. Below is the description of the CNN model for MNIST:
> A CNN with two 5x5 convolution layers (the first with 32 channels, the second with 64, each followed with 2x2 max pooling), a fully connected layer with 512 units and ReLu activation, and a final softmax output layer (1,663,370 total parameters).

Below is the description of the CNN model for CIFAR10:
> The model architecture was taken from the TensorFlow tutorial, which consists of two convolutional layers followed by two fully connected layers and then a linear transformation layer to produce logits, for a total of about 106 parameters.

The citation to which they referred in the paper cannot be found as of now. However, official TensorFlow tutorial on implementing a simple CNN can still be found [here](https://www.tensorflow.org/tutorials/images/cnn?hl=zh-cn). Alternatively, one can also refer to [this webpage](https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-tensorflow/) for the information and architecture of the example CNN.  

### III. Implementing the training

One need to take care of the training hyper-parameters that had been used in the vannila paper, e.g., batch size, epochs, learning rate and related decaying schemes:
1. To reproduce the FL training results, follow careflly what are described in the vanilla paper.
2. To reproduce the baseline training results, follow the paper and also check out how training goes like in other literatures that use MNIST and CIFAR10 (e.g., the paper *Closing the generalization gap of adaptive gradient methods in training deep neural networks*).
3. Read necessary literatures and see how to select these hyper parameters (e.g., paper *Optimal mini-batch size selection for fast gradient descent*, etc.)

### IV. Differing from the exact results in the vanilla FL paper [IMPORTANT]
Although we hope to "reproduce" the results reported in [the vanilla FL paper](https://arxiv.org/abs/1602.05629), **due to limited computational power, code resources and timeline**, one may face the following constraints for the time being:
1. not be able to initilize the model optimally or luckily with a good seed,
2. not be able to find optimized hyper parameter e.g., the learning rate, given that grid searches is very time-consuming, and thus making do with a "suitable" learning rate heuristically found.
3. have to make do with (too) earlier stopping of the training, e.g., only perform hundreds of communication rounds rather than thousands of rounds.
4. cannot afford to do the non-iid study for CIFAR10 learning task. In the vanilla FL paper, the author also did not conduct related experiments 
    * **you may report that this experiments can be carried out in the future** e.g., how to divide CIFAR10 for balanced/unbalanced non-iid scenarios.
5. not be able to perform experiment for LSTM model with language datasets

However, since our goal is to evaluate FL vs baseline methods, rather than achieving the best possible accuracy on MNIST/CIFAR10 learning tasks, therefore **simplified and reduced experiments can be acceptable as long as the results can reflect the core conclusions drawn for the comparisons reported in the vanilla paper** among FedAVg, FedSGD, and SGD, on the following aspects:
1. Increasing parallism by increasing the clients fraction does not lead to significant improvement on speed-up, with smaller batch size (B=10), a small fraction (C=0.1) is sufficient to have significant improvement over the case with C=0 (one client per round).
2. Adding more local SGD updates per round can produce a dramatic decrease in communication costs, for same local epoch E, a smaller B leads to more local mini-batch updates and a fewer rounds of communication. 
    * *To reflect this conclusion, one might not need to use that many combinations of E and B as the vanilla FL paper*.
3. FedAvg converges to a higher level of test-set accuracy than the baseline FedSGD models:
> For example, for the CNN the B = ∞, E = 1 FedSGD model eventually reaches 99.22% accuracy after 1200 rounds (and had not improved further after 6000 rounds), while the B = 10, E = 20 FedAvg model reaches an accuracy of 99.44% after 300 rounds.
4. FedAvg improves speed-up over FedSGD and baseline SGD for *test acc vs comm rounds* experiments.
    * One may skip CIFAR10 experiments if computational power does not allow for optimized learning rate searches.
    * In fact, no existing implementation has ever reproduced CIFAR10 experiments.


### V. Work logs
#### A. On CIFAR10 learning with *torch cnn* and *tf cnn* model 
##### Torch cnn only (27 April 2021)
For baseline training on CIFAR, it is found that both the *torch cnn* model created by WY and the one in AshwinRJ's repository cannot produce good test accuracy (both leads to final accuracies below 60%) after 100 epochs when a learning rate of 0.1, vanilla SGD, and a batch size of 100 is applied.

For both models, the training loss keeps reduced and then converges around 50 epochs, while the test accuracies first increase (to approximately 63%) then start to decline gradually below 60% (around 58%) only after about 15 epochs.

It looks like the models **are overfitted**.

##### Torch cnn and tf cnn (28 April 2021)
For the *torch cnn*, a fixed learning rate 0.01 does not show drops in test accuracy as was observed in the previous experiment using lr=0.1, for same epoch number 100, batch size 100 and vanilla SGD. After 100 epochs, the training loss keeps declining and test accuracy converges around 60 epochs to an accuracy about 64%.

However, for learning rate 0.01 combined with momentum 0.5/0.9, the test accuracy starts to slightly drop to around 60% after 60 epochs.

The *torch cnn* trained using η=0.01 with momentum=0.5/0.9 and the *tf cnn* trained using η=0.1 **both exhibits overfitting** after around 60 epochs, i.e., the test accuracy drops.

##### Avoid overfitting by regularization
Most simples way is the **early stopping**, which refers to the heuristic of stopping the training procedure when the error on the validation set first starts to increase. This requires the logging of test losses.

#### B. On MNIST learning with *2NN* and *CNN* models 
##### Baseline training (28 April 2021)
Both 2NN and CNN from AshwinRJ's repository are trained using vanilla SGD, η=0.01 without momentum, over 200 epochs. No overfitting ocurred, resulting a performance as below:

Model | Test acc | Time elapesed | Batch size | Epochs | Learning rate | Optimizer
------| -------- | ------------- | ---------- | ------ | ------------- | ---------
2NN   | 97.12%   | 1765s         | 100        | 200    | 0.01          | vanilla SGD 
CNN   | 99.09%   | 2014s         | 100        | 200    | 0.01          | vanilla SGD 

Both trianed models might be used for warm start in future training.

##### FedAvg training (28 April 2021)
It seems that using FedAvg on MNIST is **very slow** even with GPU: client fraction C=0.1, local epoch E=20, local batch size B=10, learning rate η=0.01 (vanilla SGD without momentum) leads to a training speed of **~40s per round, 10 rounds took 394s**.

#### C. On the optimized learning rate
##### How the vanilla FL paper did
It seems that, in the vanilla FL paper, for each set of parameter combination, the resulted learning curve shown in the figures or entry in the tables are obtained by optimizing the learning rate such that the best value of test-set accuracy was obtained. As they wrote:

>we construct a learning curve for each combination of parameter settings, optimizing η and then making each curve monotonically improving by **taking the best value of test-set accuracy achieved over all prior rounds**. We then calculate the number of rounds where the curve crosses the target accuracy...

This means that, when doing comparison, they compare the best-possible learning performance which can be obtained for each combination of parameter settings. This requires one to also conduct learning rate optimization for reproducing the exact experiment results.

The author reported that they did **grid searches** for optimizing the learning rate on *a multiplicative grid with an resolution of $10^\frac{1}{3}$ or $10^\frac{1}{6}$*.

##### Methods of grid search for the learning rate
This could be done by two stages:
1. Coarse search using a bigger resolution, e.g., a factor of 10 or 0.1, say, start from 1e-4, then 1e-3, 1e-2, 1e-1, and 1.0,
2. Finer search using a smaller resolution within an interval, i.e., between two best values found in the previous coarse search.

A grid search of optimized learning rate can be implemented using a small portion of the training dataset, **typically 20%**, this seperated portion is called **validation set** or **development set**. References includes:
* Section 4.4.4, page 95, in *Probabilistic Machine Learning: An Introduction*, 2020, by K.Murhpy
* Section 5.2, page 10, in *Stochastic gradient descent tricks*, 2012, by L.Bottou

A **source code for doing grid searches** of learning rate can be found in page 74, Chapter 4, under the section *Finding that learning rate* of the book *Programming PyTorch for Deep Learning, O'Reilly Media, 2019*

`GridSearchCV` class provided in `scikit-learn` package can also help with the grid search of learning rate, see the following resources to learn this technique:
* Page 117, section *Fine-tune Your Model/Grid Search* in chapter 2 of the book *Hands-on Machine Learning with Scikit-learn, Keras, and TensorFlow*
* [How to Grid Search Hyperparameters for Deep Learning Models in Python With Keras](https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/)
* [Hyperparameter Optimization With Random Search and Grid Search](https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/)