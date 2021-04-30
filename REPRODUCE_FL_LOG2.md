# Work logs and notes on reproducing vanilla FL paper - Part B-2: Experimental results
*NOTE: all the following results are obtained with WY's models*

### I. Baseline CIFAR10 learning with *torch cnn* and *tf cnn* model 
#### Torch cnn only
N/A

#### Torch cnn and tf cnn (28 April 2021)
N/A

### II. Baseline MNIST learning with *2NN* and *CNN* models
Model | Test acc | Time     | Batch size | Epochs | Lr     | Optim
------| -------- | -------- | ---------- | ------ | ------ | ---------
2NN   | xx.xx%   | xxxxs    | 100        | 200    | 0.??   | SGD 
CNN   | xx.xx%   | xxxxs    | 100        | 200    | 0.??   | SGD 

### III. FedAvg MNIST learning with *2NN* and *CNN* models 
#### A. Experiment 1: increase parallism
* The runs using optimized learning rate will be marked as "0.01/o"
* H Rnd means the round number where test acc hit the target, i.e, 98% for CNN, 96% for 2NN
* T Rnd means the total number of performed rounds

##### CNN/IID
Model |Method|Data  | Test acc |H Rnd |T Rnd |Time      | Machine | Frac | E | B | Lr    | Optim | Status
------|------|------| -------- |----- |----- |--------  |-------- | -----|---|---| ----- | ----- | ------
CNN   |FedAVg|iid   | 98.22%   |76    |100   |0.07hrs   | A       | 0.0  |5  |10 | 0.01  | SGD   | 
CNN   |FedAVg|iid   | 98.6%    |40    |100   |3.6hrs    | A       | 1.0  |5  |10 | 0.01  | SGD   | 
CNN   |FedAVg|iid   | 98.6%    |47    |100   |1.7hrs    | A       | 0.5  |5  |10 | 0.01  | SGD   | 
CNN   |FedAVg|iid   | 98.7%    |44    |100   |0.7hrs    | A       | 0.2  |5  |10 | 0.01  | SGD   | 
CNN   |FedAVg|iid   | 98.6%    |52    |100   |xxxhrs    | A       | 0.1  |5  |10 | 0.01  | SGD   | 
CNN   |FedAVg|iid   | 96.91%   |xx    |1000  |0.52hrs   | A       | 0.0  |5  |∞  | 0.01  | SGD   | 
CNN   |FedAVg|iid   | xxxx%    |xx    |100   |xxxhrs    | A       | 1.0  |5  |∞  | 0.01  | SGD   |
CNN   |FedAVg|iid   | xxxx%    |xx    |100   |xxxhrs    | A       | 0.5  |5  |∞  | 0.01  | SGD   |
CNN   |FedAVg|iid   | xxxx%    |xx    |100   |xxxhrs    | A       | 0.2  |5  |∞  | 0.01  | SGD   |
CNN   |FedAVg|iid   | xxxx%    |xx    |100   |xxxhrs    | A       | 0.1  |5  |∞  | 0.01  | SGD   |

##### Remarks
1. Set 96%, 98% as targets for CNN, 2NN is to avoid the needed rounds from being too large to complete in time.
2. For non-IID cases, one may need to use an even lower target to avoid too long running time.
3. The time taken for C=1.0 is formidable, even for IID cases. So, one could consider 200 rounds for C=1.0, E=5, B=10 of non-IID, in order to complete in allowed timeline.

##### CNN/non-IID
Model |Method|Data  | Test acc |H Rnd |T Rnd |Time      | Machine | Frac | E | B | Lr    | Optim | Status
------|------|------| -------- |----- |----- |--------  |-------- | -----|---|---| ----- | ----- | ------
CNN   |FedAVg|N-iid | xxxx%    |xx    |100   |xxxhrs    | A       | 0.0  |5  |10 | 0.01  | SGD   |
CNN   |FedAVg|N-iid | xxxx%    |xx    |100   |xxxhrs    | A       | 1.0  |5  |10 | 0.01  | SGD   |
CNN   |FedAVg|N-iid | xxxx%    |xx    |100   |xxxhrs    | A       | 0.5  |5  |10 | 0.01  | SGD   |
CNN   |FedAVg|N-iid | xxxx%    |xx    |100   |xxxhrs    | A       | 0.2  |5  |10 | 0.01  | SGD   |
CNN   |FedAVg|N-iid | xxxx%    |xx    |100   |xxxhrs    | A       | 0.1  |5  |10 | 0.01  | SGD   |
CNN   |FedAVg|N-iid | xxxx%    |xx    |100   |xxxhrs    | A       | 0.0  |5  |∞  | 0.01  | SGD   |
CNN   |FedAVg|N-iid | xxxx%    |xx    |100   |xxxhrs    | A       | 1.0  |5  |∞  | 0.01  | SGD   |
CNN   |FedAVg|N-iid | xxxx%    |xx    |100   |xxxhrs    | A       | 0.5  |5  |∞  | 0.01  | SGD   |
CNN   |FedAVg|N-iid | xxxx%    |xx    |100   |xxxhrs    | A       | 0.2  |5  |∞  | 0.01  | SGD   |
CNN   |FedAVg|N-iid | xxxx%    |xx    |100   |xxxhrs    | A       | 0.1  |5  |∞  | 0.01  | SGD   |
CNN   |FedAVg|N-iid | xxxx%    |xx    |100   |xxxhrs    | A       | 0.0  |5  |∞  | 0.01  | SGD   |

##### 2NN/IID
Model |Method|Data  | Test acc |H Rnd |T Rnd |Time      | Machine | Frac | E | B | Lr    | Optim | Status
------|------|------| -------- |----- |----- |--------  |-------- | -----|---|---| ----- | ----- | ------
2NN   |FedAVg|iid   | xxxx%    |xx    |100   |xxxhrs    | A       | 0.0  |5  |10 | 0.01  | SGD   |
2NN   |FedAVg|iid   | xxxx%    |xx    |100   |xxxhrs    | A       | 1.0  |5  |10 | 0.01  | SGD   | 
2NN   |FedAVg|iid   | xxxx%    |xx    |100   |xxxhrs    | A       | 0.5  |5  |10 | 0.01  | SGD   |
2NN   |FedAVg|iid   | xxxx%    |xx    |100   |xxxhrs    | A       | 0.2  |5  |10 | 0.01  | SGD   |
2NN   |FedAVg|iid   | xxxx%    |xx    |100   |xxxhrs    | A       | 0.1  |5  |10 | 0.01  | SGD   |
2NN   |FedAVg|iid   | xxxx%    |xx    |100   |xxxhrs    | A       | 0.0  |5  |∞  | 0.01  | SGD   |
2NN   |FedAVg|iid   | xxxx%    |xx    |100   |xxxhrs    | A       | 1.0  |5  |∞  | 0.01  | SGD   |
2NN   |FedAVg|iid   | xxxx%    |xx    |100   |xxxhrs    | A       | 0.5  |5  |∞  | 0.01  | SGD   |
2NN   |FedAVg|iid   | xxxx%    |xx    |100   |xxxhrs    | A       | 0.2  |5  |∞  | 0.01  | SGD   |
2NN   |FedAVg|iid   | xxxx%    |xx    |100   |xxxhrs    | A       | 0.1  |5  |∞  | 0.01  | SGD   |

##### 2NN/non-IID
Model |Method|Data  | Test acc |H Rnd |T Rnd |Time      | Machine | Frac | E | B | Lr    | Optim | Status
------|------|------| -------- |----- |----- |--------  |-------- | -----|---|---| ----- | ----- | ------
2NN   |FedAVg|N-iid | xxxx%    |xx    |100   |xxxhrs    | A       | 0.0  |5  |10 | 0.01  | SGD   |
2NN   |FedAVg|N-iid | xxxx%    |xx    |100   |xxxhrs    | A       | 1.0  |5  |10 | 0.01  | SGD   | 
2NN   |FedAVg|N-iid | xxxx%    |xx    |100   |xxxhrs    | A       | 0.5  |5  |10 | 0.01  | SGD   |
2NN   |FedAVg|N-iid | xxxx%    |xx    |100   |xxxhrs    | A       | 0.2  |5  |10 | 0.01  | SGD   |
2NN   |FedAVg|N-iid | xxxx%    |xx    |100   |xxxhrs    | A       | 0.1  |5  |10 | 0.01  | SGD   |
2NN   |FedAVg|N-iid | xxxx%    |xx    |100   |xxxhrs    | A       | 0.0  |5  |∞  | 0.01  | SGD   |
2NN   |FedAVg|N-iid | xxxx%    |xx    |100   |xxxhrs    | A       | 1.0  |5  |∞  | 0.01  | SGD   |
2NN   |FedAVg|N-iid | xxxx%    |xx    |100   |xxxhrs    | A       | 0.5  |5  |∞  | 0.01  | SGD   |
2NN   |FedAVg|N-iid | xxxx%    |xx    |100   |xxxhrs    | A       | 0.2  |5  |∞  | 0.01  | SGD   |
2NN   |FedAVg|N-iid | xxxx%    |xx    |100   |xxxhrs    | A       | 0.1  |5  |∞  | 0.01  | SGD   |



#### B. Experiment 2: increase local computation
* The fraction number is fixed at C=0.1
* The runs using optimized learning rate will be marked as "0.01/o"
* T Rnd means the total number of performed rounds

##### CNN/IID
Model |Method|Data  | Test acc   |T Rnd |Time      | Machine | Frac | E | B | Lr    | Optim | Status
------|------|------| --------   |----  |--------  |-------- | -----|---|---| ----- | ------| -----
CNN   |FedSGD|iid   | xxxxx%     |1000  |0.65hrs   | T       | 0.1  |1  |∞  | 0.08  | SGD   | 
CNN   |FedAVg|iid   | xxxxx%     |1000  |1.70hrs   | T       | 0.1  |5  |∞  | 0.01  | SGD   | 
CNN   |FedAVg|iid   | xxxxx%     |1000  |0.76hrs   | T       | 0.1  |1  |50 | 0.10  | SGD   | 
CNN   |FedAVg|iid   | 99.28%     |1000  |6.04hrs   | T       | 0.1  |20 |∞  | 0.2   | SGD   | done 
CNN   |FedAVg|iid   | xxxxx%     |1000  |0.88hrs   | T       | 0.1  |1  |10 | 0.01  | SGD   | 
CNN   |FedAVg|iid   | %          |xxxx  |hrs       | T       | 0.1  |5  |50 | 0.01  | SGD   | 
CNN   |FedAVg|iid   | %          |xxxx  |hrs       | T       | 0.1  |20 |50 | 0.1   | SGD   | ...
CNN   |FedAVg|iid   | %          |xxxx  |hrs       | T       | 0.1  |5  |10 | 0.01  | SGD   | 
CNN   |FedAVg|iid   | xxxxx%     |1000  |8.0hrs    | T       | 0.1  |20 |10 | 0.01  | SGD   | 
##### Remarks
1. For E=1, B=inf, after 1000 rounds, the test acc can still be improve much, the training loss can also be further reduced. It seems that either 1000 rounds is not enough or the learning rate needs optimized.
2. An make-do approach under the limited computational power, is to **use the same, non-optimized lr, with a lower target (e.g., 91%) for benchmarking the speed up for ever set of parameter combinations against the FedSGD**.
3. 400-500 rounds should be sufficient for E=5 B=50

##### CNN/non-IID
Model |Method|Data  | Test acc   |T Rnd |Time      | Machine | Frac | E | B | Lr    | Optim | Status
------|------|------| --------   |----  |--------  |-------- | -----|---|---| ----- | ------| -----
CNN   |FedSGD|N-iid | %          |xxxx  |    hrs   | T       | 0.1  |1  |∞  | 0.01  | SGD   | 
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
2NN   |FedSGD|N-iid | %          |xxxx  |   hrs    | T       | 0.1  |1  |∞  | 0.01  | SGD   |
2NN   |FedAVg|N-iid | %          |xxxx  |hrs       | T       | 0.1  |10 |∞  | 0.01  | SGD   | 
2NN   |FedAVg|N-iid | %          |xxxx  |hrs       | T       | 0.1  |1  |50 | 0.01  | SGD   | 
2NN   |FedAVg|N-iid | %          |xxxx  |hrs       | T       | 0.1  |20 |∞  | 0.01  | SGD   | 
2NN   |FedAVg|N-iid | %          |xxxx  |hrs       | T       | 0.1  |1  |10 | 0.01  | SGD   | 
2NN   |FedAVg|N-iid | %          |xxxx  |hrs       | T       | 0.1  |10 |50 | 0.01  | SGD   | 
2NN   |FedAVg|N-iid | %          |xxxx  |hrs       | T       | 0.1  |20 |50 | 0.01  | SGD   | 
2NN   |FedAVg|N-iid | %          |xxxx  |hrs       | T       | 0.1  |10 |10 | 0.01  | SGD   | 
2NN   |FedAvg|N-iid | 96.4%      |1000  |8.0hrs    | T       | 0.1  |20 |10 | 0.01  | SGD   | done

#### C. Training time summary
* The runs using optimized learning rate will be marked as "0.01-o"

Model |Method| Data |Time/rnd | 100-rnd time    | Machine |Frac | E | B | Lr    | Optim
------|------|------|-------- | --------------  |-------- |-----|---|---| ----- | ---------
CNN   |FedSGD|M-iid | 2.2s    | 3.7mins/0.06hrs | T       |0.1  |1  | ∞ | 0.01  | SGD
CNN   |FedAVg|M-iid | 5.6s    | 9.3mins/0.16hrs | T       |0.1  |5  | ∞ | 0.01  | SGD
CNN   |FedAvg|M-iid | 129.12s | 3.6hrs          | A       |1.0  |5  |10 | 0.01  | SGD 
CNN   |FedAvg|M-iid | 39.4s   | 1.1hrs          | A       |0.1  |20 |10 | 0.01  | SGD
CNN   |FedAvg|M-iid | 25.2s   | 42mins/0.7hrs   | A       |0.2  |5  |10 | 0.01  | SGD
CNN   |FedAvg|M-iid | 42.34s  | 1.2hrs          | T       |0.1  |20 |10 | 0.01  | SGD
CNN   |FedAvg|M-iid | 2.2s    | 3.6mins/0.06hrs | T       |0.1  |1  |50 | 0.01  | SGD


It seems that FedAvg for the IID data is **very slow** even with GPU, not to mention the non-IID cases.
