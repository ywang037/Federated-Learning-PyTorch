# On reproducing the results in classic FL paper

### V. Work logs and notes
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

##### FedAvg training

##### *Training results*

Model | Test acc   | Time elapsed  | Machine | Frac | Local B | Local E | Learning rate | Optimizer
------| --------   | ------------  |-------- | -----| ------- | ------  | ------------- | ---------
2NN   | 129.12s    | 3.6hrs        | Acer    | 1.0  | 10      | 5       | 0.01          | SGD 


It seems that FedAvg for the IID data is **very slow** even with GPU, not to mention the non-IID cases.
##### *Training time records*

Model | Time/round | 100-round time  | Machine | Frac | Local B | Local E | Learning rate | Optimizer
------| --------   | --------------  |-------- | -----| ------- | ------  | ------------- | ---------
CNN   | 129.12s    | 3.6hrs          | Acer    | 1.0  | 10      | 5       | 0.01          | SGD 
CNN   | 39.4  s    | 1.1hrs          | Acer    | 0.1  | 10      | 20      | 0.01          | SGD 
CNN   | 42.34s     | 1.2hrs          | Think   | 0.1  | 10      | 20      | 0.01          | SGD
CNN   | 2.2s       | 3.7mins/0.06hrs | Think   | 0.1  | ∞       | 1       | 0.01          | SGD
