# Work logs and notes on reproducing vanilla FL paper

### I. Baseline CIFAR10 learning with *torch cnn* and *tf cnn* model 
#### Torch cnn only (27 April 2021)
For baseline training on CIFAR, it is found that both the *torch cnn* model created by WY and the one in AshwinRJ's repository cannot produce good test accuracy (both leads to final accuracies below 60%) after 100 epochs when a learning rate of 0.1, vanilla SGD, and a batch size of 100 is applied.

For both models, the training loss keeps reduced and then converges around 50 epochs, while the test accuracies first increase (to approximately 63%) then start to decline gradually below 60% (around 58%) only after about 15 epochs.

It looks like the models **are overfitted**.

#### Torch cnn and tf cnn (28 April 2021)
For the *torch cnn*, a fixed learning rate 0.01 does not show drops in test accuracy as was observed in the previous experiment using lr=0.1, for same epoch number 100, batch size 100 and vanilla SGD. After 100 epochs, the training loss keeps declining and test accuracy converges around 60 epochs to an accuracy about 64%.

However, for learning rate 0.01 combined with momentum 0.5/0.9, the test accuracy starts to slightly drop to around 60% after 60 epochs.

The *torch cnn* trained using η=0.01 with momentum=0.5/0.9 and the *tf cnn* trained using η=0.1 **both exhibits overfitting** after around 60 epochs, i.e., the test accuracy drops.

#### Avoid overfitting by regularization
Most simples way is the **early stopping**, which refers to the heuristic of stopping the training procedure when the error on the validation set first starts to increase. This requires the logging of test losses.

### II. Baseline MNIST learning with *2NN* and *CNN* models (28 April 2021)
Both 2NN and CNN from AshwinRJ's repository are trained using vanilla SGD, η=0.01 without momentum, over 200 epochs. No overfitting ocurred, resulting a performance as below:

Model | Test acc | Time     | Batch size | Epochs | Lr     | Optim
------| -------- | -------- | ---------- | ------ | ------ | ---------
2NN   | 97.12%   | 1765s    | 100        | 200    | 0.01   | vanilla SGD 
CNN   | 99.09%   | 2014s    | 100        | 200    | 0.01   | vanilla SGD 

Both trianed models might be used for warm start in future training.

### III. FedAvg MNIST learning with *2NN* and *CNN* models 
*NOTE: all the following results are obtained with **non-optimized** learning rate*
#### A. Experiment 1: increase parallism
* The runs using optimized learning rate will be marked as "0.01-o"
* H Rnd means the round number where test acc hit the target, i.e, 98% for CNN, 96% for 2NN
* T Rnd means the total number of performed rounds

##### CNN/IID

Model |Method|Data  | Test acc |H Rnd |T Rnd |Time      | Machine | Frac | E | B | Lr    | Optim | Status
------|------|------| -------- |----- |----- |--------  |-------- | -----|---|---| ----- | ----- | ------
CNN   |FedAVg|iid   | 98.6%    |40    |100   |3.6hrs    | A       | 1.0  |5  |10 | 0.01  | SGD   | done
CNN   |FedAVg|iid   | xxxx%    |xx    |100   |xxxhrs    | A       | 0.5  |5  |10 | 0.01  | SGD   |
CNN   |FedAVg|iid   | xxxx%    |xx    |100   |xxxhrs    | A       | 0.2  |5  |10 | 0.01  | SGD   |
CNN   |FedAVg|iid   | xxxx%    |xx    |100   |xxxhrs    | A       | 0.1  |5  |10 | 0.01  | SGD   |
CNN   |FedAVg|iid   | xxxx%    |xx    |100   |xxxhrs    | A       | 0.0  |5  |10 | 0.01  | SGD   |
CNN   |FedAVg|iid   | xxxx%    |xx    |100   |xxxhrs    | A       | 1.0  |5  |∞  | 0.01  | SGD   |
CNN   |FedAVg|iid   | xxxx%    |xx    |100   |xxxhrs    | A       | 0.5  |5  |∞  | 0.01  | SGD   |
CNN   |FedAVg|iid   | xxxx%    |xx    |100   |xxxhrs    | A       | 0.2  |5  |∞  | 0.01  | SGD   |
CNN   |FedAVg|iid   | xxxx%    |xx    |100   |xxxhrs    | A       | 0.1  |5  |∞  | 0.01  | SGD   |
CNN   |FedAVg|iid   | xxxx%    |xx    |100   |xxxhrs    | A       | 0.0  |5  |∞  | 0.01  | SGD   |
##### Remarks
1. Set 96%, 98% as targets for CNN, 2NN is to avoid the needed rounds from being too large to complete in time.
2. For non-IID cases, one may need to use an even lower target to avoid too long running time.
3. The time taken for C=1.0 is formidable, even for IID cases. So, one could consider 200 rounds for C=1.0, E=5, B=10 of non-IID, in order to complete in allowed timeline.

##### CNN/non-IID
Model |Method|Data  | Test acc |H Rnd |T Rnd |Time      | Machine | Frac | E | B | Lr    | Optim | Status
------|------|------| -------- |----- |----- |--------  |-------- | -----|---|---| ----- | ----- | ------
CNN   |FedAVg|N-iid | 98.6%    |40    |100   |3.6hrs    | A       | 1.0  |5  |10 | 0.01  | SGD   | done
CNN   |FedAVg|N-iid | xxxx%    |xx    |100   |xxxhrs    | A       | 0.5  |5  |10 | 0.01  | SGD   |
CNN   |FedAVg|N-iid | xxxx%    |xx    |100   |xxxhrs    | A       | 0.2  |5  |10 | 0.01  | SGD   |
CNN   |FedAVg|N-iid | xxxx%    |xx    |100   |xxxhrs    | A       | 0.1  |5  |10 | 0.01  | SGD   |
CNN   |FedAVg|N-iid | xxxx%    |xx    |100   |xxxhrs    | A       | 0.0  |5  |10 | 0.01  | SGD   |
CNN   |FedAVg|N-iid | xxxx%    |xx    |100   |xxxhrs    | A       | 1.0  |5  |∞  | 0.01  | SGD   |
CNN   |FedAVg|N-iid | xxxx%    |xx    |100   |xxxhrs    | A       | 0.5  |5  |∞  | 0.01  | SGD   |
CNN   |FedAVg|N-iid | xxxx%    |xx    |100   |xxxhrs    | A       | 0.2  |5  |∞  | 0.01  | SGD   |
CNN   |FedAVg|N-iid | xxxx%    |xx    |100   |xxxhrs    | A       | 0.1  |5  |∞  | 0.01  | SGD   |
CNN   |FedAVg|N-iid | xxxx%    |xx    |100   |xxxhrs    | A       | 0.0  |5  |∞  | 0.01  | SGD   |

##### 2NN/IID
Model |Method|Data  | Test acc |H Rnd |T Rnd |Time      | Machine | Frac | E | B | Lr    | Optim | Status
------|------|------| -------- |----- |----- |--------  |-------- | -----|---|---| ----- | ----- | ------
2NN   |FedAVg|iid   | xxxx%    |xx    |100   |xxxhrs    | A       | 1.0  |5  |10 | 0.01  | SGD   | 
2NN   |FedAVg|iid   | xxxx%    |xx    |100   |xxxhrs    | A       | 0.5  |5  |10 | 0.01  | SGD   |
2NN   |FedAVg|iid   | xxxx%    |xx    |100   |xxxhrs    | A       | 0.2  |5  |10 | 0.01  | SGD   |
2NN   |FedAVg|iid   | xxxx%    |xx    |100   |xxxhrs    | A       | 0.1  |5  |10 | 0.01  | SGD   |
2NN   |FedAVg|iid   | xxxx%    |xx    |100   |xxxhrs    | A       | 0.0  |5  |10 | 0.01  | SGD   |
2NN   |FedAVg|iid   | xxxx%    |xx    |100   |xxxhrs    | A       | 1.0  |5  |∞  | 0.01  | SGD   |
2NN   |FedAVg|iid   | xxxx%    |xx    |100   |xxxhrs    | A       | 0.5  |5  |∞  | 0.01  | SGD   |
2NN   |FedAVg|iid   | xxxx%    |xx    |100   |xxxhrs    | A       | 0.2  |5  |∞  | 0.01  | SGD   |
2NN   |FedAVg|iid   | xxxx%    |xx    |100   |xxxhrs    | A       | 0.1  |5  |∞  | 0.01  | SGD   |
2NN   |FedAVg|iid   | xxxx%    |xx    |100   |xxxhrs    | A       | 0.0  |5  |∞  | 0.01  | SGD   |

##### 2NN/non-IID
Model |Method|Data  | Test acc |H Rnd |T Rnd |Time      | Machine | Frac | E | B | Lr    | Optim | Status
------|------|------| -------- |----- |----- |--------  |-------- | -----|---|---| ----- | ----- | ------
2NN   |FedAVg|iid   | xxxx%    |xx    |100   |xxxhrs    | A       | 1.0  |5  |10 | 0.01  | SGD   | 
2NN   |FedAVg|N-iid | xxxx%    |xx    |100   |xxxhrs    | A       | 0.5  |5  |10 | 0.01  | SGD   |
2NN   |FedAVg|N-iid | xxxx%    |xx    |100   |xxxhrs    | A       | 0.2  |5  |10 | 0.01  | SGD   |
2NN   |FedAVg|N-iid | xxxx%    |xx    |100   |xxxhrs    | A       | 0.1  |5  |10 | 0.01  | SGD   |
2NN   |FedAVg|N-iid | xxxx%    |xx    |100   |xxxhrs    | A       | 0.0  |5  |10 | 0.01  | SGD   |
2NN   |FedAVg|N-iid | xxxx%    |xx    |100   |xxxhrs    | A       | 1.0  |5  |∞  | 0.01  | SGD   |
2NN   |FedAVg|N-iid | xxxx%    |xx    |100   |xxxhrs    | A       | 0.5  |5  |∞  | 0.01  | SGD   |
2NN   |FedAVg|N-iid | xxxx%    |xx    |100   |xxxhrs    | A       | 0.2  |5  |∞  | 0.01  | SGD   |
2NN   |FedAVg|N-iid | xxxx%    |xx    |100   |xxxhrs    | A       | 0.1  |5  |∞  | 0.01  | SGD   |
2NN   |FedAVg|N-iid | xxxx%    |xx    |100   |xxxhrs    | A       | 0.0  |5  |∞  | 0.01  | SGD   |


#### B. Experiment 2: increase local computation
* The fraction number is fixed at C=0.1
* The runs using optimized learning rate will be marked as "0.01-o"
* T Rnd means the total number of performed rounds

##### Remarks
1. For E=1, B=inf, after 1000 rounds, the test acc can still be improve much, the training loss can also be further reduced. It seems that either 1000 rounds is not enough or the learning rate needs optimized.
2. An make-do approach under the limited computational power, is to **use the same, non-optimized lr, with a lower target (e.g., 91%) for benchmarking the speed up for ever set of parameter combinations against the FedSGD**.

##### CNN/IID
Model |Method|Data  | Test acc   |T Rnd |Time      | Machine | Frac | E | B | Lr    | Optim | Status
------|------|------| --------   |----  |--------  |-------- | -----|---|---| ----- | ------| -----
CNN   |FedSGD|iid   | 91.99%     |1000  |0.56hrs   | T       | 0.1  |1  |∞  | 0.01  | SGD   | done
CNN   |FedAVg|iid   | %          |xxxx  |hrs       | T       | 0.1  |5  |∞  | 0.01  | SGD   | 
CNN   |FedAVg|iid   | %          |xxxx  |hrs       | T       | 0.1  |1  |50 | 0.01  | SGD   | 
CNN   |FedAVg|iid   | %          |xxxx  |hrs       | T       | 0.1  |20 |∞  | 0.01  | SGD   | 
CNN   |FedAVg|iid   | %          |xxxx  |hrs       | T       | 0.1  |1  |10 | 0.01  | SGD   | 
CNN   |FedAVg|iid   | %          |xxxx  |hrs       | T       | 0.1  |5  |50 | 0.01  | SGD   | 
CNN   |FedAVg|iid   | %          |xxxx  |hrs       | T       | 0.1  |20 |50 | 0.01  | SGD   | 
CNN   |FedAVg|iid   | %          |xxxx  |hrs       | T       | 0.1  |5  |10 | 0.01  | SGD   | 
CNN   |FedAVg|iid   | 96.4%      |1000  |8.0hrs    | T       | 0.1  |20 |10 | 0.01  | SGD   | done

##### CNN/non-IID
Model |Method|Data  | Test acc   |T Rnd |Time      | Machine | Frac | E | B | Lr    | Optim | Status
------|------|------| --------   |----  |--------  |-------- | -----|---|---| ----- | ------| -----
CNN   |FedSGD|N-iid | 91.99%     |1000  |0.56hrs   | T       | 0.1  |1  |∞  | 0.01  | SGD   | done
CNN   |FedAVg|N-iid | %          |xxxx  |hrs       | T       | 0.1  |5  |∞  | 0.01  | SGD   | 
CNN   |FedAVg|N-iid | %          |xxxx  |hrs       | T       | 0.1  |1  |50 | 0.01  | SGD   | 
CNN   |FedAVg|N-iid | %          |xxxx  |hrs       | T       | 0.1  |20 |∞  | 0.01  | SGD   | 
CNN   |FedAVg|N-iid | %          |xxxx  |hrs       | T       | 0.1  |1  |10 | 0.01  | SGD   | 
CNN   |FedAVg|N-iid | %          |xxxx  |hrs       | T       | 0.1  |5  |50 | 0.01  | SGD   | 
CNN   |FedAVg|N-iid | %          |xxxx  |hrs       | T       | 0.1  |20 |50 | 0.01  | SGD   | 
CNN   |FedAVg|N-iid | %          |xxxx  |hrs       | T       | 0.1  |5  |10 | 0.01  | SGD   | 
CNN   |FedAVg|N-iid | 96.4%      |1000  |8.0hrs    | T       | 0.1  |20 |10 | 0.01  | SGD   | done

##### 2NN/IID
Model |Method|Data  | Test acc   |T Rnd |Time      | Machine | Frac | E | B | Lr    | Optim | Stauts
------|------|------| --------   |----  |--------  |-------- | -----|---|---| ----- | ------| ---
2NN   |FedSGD|iid   | %          |xxxx  |   hrs    | T       | 0.1  |1  |∞  | 0.01  | SGD   |
2NN   |FedAVg|iid   | %          |xxxx  |hrs       | T       | 0.1  |10 |∞  | 0.01  | SGD   | 
2NN   |FedAVg|iid   | %          |xxxx  |hrs       | T       | 0.1  |1  |50 | 0.01  | SGD   | 
2NN   |FedAVg|iid   | %          |xxxx  |hrs       | T       | 0.1  |20 |∞  | 0.01  | SGD   | 
2NN   |FedAVg|iid   | %          |xxxx  |hrs       | T       | 0.1  |1  |10 | 0.01  | SGD   | 
2NN   |FedAVg|iid   | %          |xxxx  |hrs       | T       | 0.1  |10 |50 | 0.01  | SGD   | 
2NN   |FedAVg|iid   | %          |xxxx  |hrs       | T       | 0.1  |20 |50 | 0.01  | SGD   | 
2NN   |FedAVg|iid   | %          |xxxx  |hrs       | T       | 0.1  |10 |10 | 0.01  | SGD   | 
2NN   |FedAvg|iid   | 96.4%      |1000  |8.0hrs    | T       | 0.1  |20 |10 | 0.01  | SGD   | done

##### 2NN/non-IID
Model |Method|Data  | Test acc   |T Rnd |Time      | Machine | Frac | E | B | Lr    | Optim | Stauts
------|------|------| --------   |----  |--------  |-------- | -----|---|---| ----- | ------| ---
2NN   |FedSGD|M-iid | %          |xxxx  |   hrs    | T       | 0.1  |1  |∞  | 0.01  | SGD   |
2NN   |FedAVg|M-iid | %          |xxxx  |hrs       | T       | 0.1  |10 |∞  | 0.01  | SGD   | 
2NN   |FedAVg|M-iid | %          |xxxx  |hrs       | T       | 0.1  |1  |50 | 0.01  | SGD   | 
2NN   |FedAVg|M-iid | %          |xxxx  |hrs       | T       | 0.1  |20 |∞  | 0.01  | SGD   | 
2NN   |FedAVg|M-iid | %          |xxxx  |hrs       | T       | 0.1  |1  |10 | 0.01  | SGD   | 
2NN   |FedAVg|M-iid | %          |xxxx  |hrs       | T       | 0.1  |10 |50 | 0.01  | SGD   | 
2NN   |FedAVg|M-iid | %          |xxxx  |hrs       | T       | 0.1  |20 |50 | 0.01  | SGD   | 
2NN   |FedAVg|M-iid | %          |xxxx  |hrs       | T       | 0.1  |10 |10 | 0.01  | SGD   | 
2NN   |FedAvg|M-iid | 96.4%      |1000  |8.0hrs    | T       | 0.1  |20 |10 | 0.01  | SGD   | done

#### C. Training time summary
* The runs using optimized learning rate will be marked as "0.01-o"

Model |Method| Data |Time/rnd | 100-rnd time    | Machine |Frac | E | B | Lr    | Optim
------|------|------|-------- | --------------  |-------- |-----|---|---| ----- | ---------
CNN   |FedSGD|M-iid | 2.2s    | 3.7mins/0.06hrs | T       |0.1  |1  | ∞ | 0.01  | SGD
CNN   |FedSGD|M-iid | ~6.0s   | mins/       hrs | T       |0.1  |5  | ∞ | 0.01  | SGD
CNN   |FedAvg|M-iid | 129.12s | 3.6hrs          | A       |1.0  |5  |10 | 0.01  | SGD 
CNN   |FedAvg|M-iid | 39.4s   | 1.1hrs          | A       |0.1  |20 |10 | 0.01  | SGD 
CNN   |FedAvg|M-iid | 42.34s  | 1.2hrs          | T       |0.1  |20 |10 | 0.01  | SGD


It seems that FedAvg for the IID data is **very slow** even with GPU, not to mention the non-IID cases.
