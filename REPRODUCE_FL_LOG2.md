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
* R-98 means the number of round where test acc hit 98%, similar for R-XX
* T Rnd means the total number of performed rounds
* Test acc (f,max) means final value and max value 

##### CNN/IID
Model |Method|Data  | Test acc (f,max) |R-98  |T Rnd |Time      | Machine | Frac | E | B | Lr     | Optim | Status
------|------|------| --------         |----- |----- |--------  |-------- | -----|---|---| -----  | ----- | ------
CNN   |FedAVg|iid   | 98.58%,99.23%    |20    |500   |0.42hrs   | A       | 0.0  |5  |10 | 0.1/o  | SGD   | done
CNN   |FedAVg|iid   |                  |      |      |          | A       | 1.0  |5  |10 |        | SGD   | cancelled
CNN   |FedAVg|iid   | xxxx%            |5     |100   |xxxxhrs   | A       | 0.5  |5  |10 | 0.1/o  | SGD   | run on T
CNN   |FedAVg|iid   | 99.35%,99.36%    |5     |100   |0.69hrs   | A       | 0.2  |5  |10 | 0.1/o  | SGD   | done
CNN   |FedAVg|iid   | 99.36%,99.36%    |6     |100   |0.38hrs   | A       | 0.1  |5  |10 | 0.1/o  | SGD   | done
CNN   |FedAVg|iid   | 98.19%,98.45%    |88    |100   |0.05hrs   | A       | 0.0  |5  |∞  | 0.1/o  | SGD   | done
CNN   |FedAVg|iid   |                  |      |100   |          | A       | 1.0  |5  |∞  |        | SGD   | cancelled
CNN   |FedAVg|iid   | xxxx%            |xx    |100   |xxxhrs    | A       | 0.5  |5  |∞  | 0.2/o  | SGD   | run on T
CNN   |FedAVg|iid   | 98.66%,98.70%    |54    |100   |0.32hrs   | T       | 0.2  |5  |∞  | 0.2/o  | SGD   | done
CNN   |FedAVg|iid   | 98.36%,98.37%    |67    |100   |0.18hrs   | T       | 0.1  |5  |∞  | 0.2/o  | SGD   | done

##### Remarks
1. Set 96%, 98% as targets for CNN, 2NN is to avoid the needed rounds from being too large to complete in time.
2. For non-IID cases, one may need to use an even lower target to avoid too long running time.
3. The time taken for C=1.0 is formidable, even for IID cases. So, one could consider 200 rounds for C=1.0, E=5, B=10 of non-IID, in order to complete in allowed timeline.

##### CNN/non-IID
Model |Method|Data  | Test acc (f,max) |R-98  |T Rnd |Time      | Machine | Frac | E | B | Lr     | Optim | Status
------|------|------| --------         |----- |----- |--------  |-------- | -----|---|---| -----  | ----- | ------
CNN   |FedAVg|N-iid | 99.27%,99.38%    |416   |1500  |1.4hrs    | A       | 0.0  |5  |10 | 0.04/o | SGD   | done
CNN   |FedAVg|N-iid |                  |      |      |          | A       | 1.0  |5  |10 |        | SGD   | cancelled
CNN   |FedAVg|N-iid | xxxx%            |xx    |200   |xxxhrs    | A       | 0.5  |5  |10 | 0.04/o | SGD   | to run on A
CNN   |FedAVg|N-iid | 99.11%,99.19%    |40    |250   |1.67hrs   | A       | 0.2  |5  |10 | 0.04/o | SGD   | done
CNN   |FedAVg|N-iid | 99.21%,99.30%    |75    |1500  |6.25hrs   | A       | 0.1  |5  |10 | 0.04/o | SGD   | done
CNN   |FedAVg|N-iid | 98.62%,98.98%    |778   |1500  |1.00hrs   | A       | 0.0  |5  |∞  | 0.02/o | SGD   | done
CNN   |FedAVg|N-iid |                  |      |      |          | A       | 1.0  |5  |∞  |        | SGD   | cancelled
CNN   |FedAVg|N-iid | xxxx%            |xx    |200   |xxxhrs    | A       | 0.5  |5  |∞  | 0.04/o | SGD   | to run on A
CNN   |FedAVg|N-iid | 98.99%,99.08%    |319   |1500  |5.54rs    | A       | 0.2  |5  |∞  | 0.04/o | SGD   | done
CNN   |FedAVg|N-iid | 98.92%,99.07%    |323   |1500  |3.08hrs   | A       | 0.1  |5  |∞  | 0.04/o | SGD   | done
##### Remarks
1. Non-IID, {E=5, B=∞}  across 98% earlier than 500 rounds, and may not be albe to reach 99%
2. The larger the fraction, the quicker the convergence, since the learning rate are almost the same, so that for C=0.5, one may use fewer rounds, e.g., 500-1000 

##### 2NN/IID
Model |Method|Data  | Test acc (f,max) |R-98  |T Rnd |Time      | Machine | Frac | E | B | Lr     | Optim | Status
------|------|------| --------         |----- |----- |--------  |-------- | -----|---|---| -----  | ----- | ------
2NN   |FedAVg|iid   | xxxx%            |xx    |100   |xxxhrs    | A       | 0.0  |5  |10 |        | SGD   |
2NN   |FedAVg|iid   | xxxx%            |xx    |100   |xxxhrs    | A       | 1.0  |5  |10 |        | SGD   | 
2NN   |FedAVg|iid   | xxxx%            |xx    |100   |xxxhrs    | A       | 0.5  |5  |10 |        | SGD   |
2NN   |FedAVg|iid   | xxxx%            |xx    |100   |xxxhrs    | A       | 0.2  |5  |10 |        | SGD   |
2NN   |FedAVg|iid   | xxxx%            |xx    |100   |xxxhrs    | A       | 0.1  |5  |10 |        | SGD   |
2NN   |FedAVg|iid   | xxxx%            |xx    |100   |xxxhrs    | A       | 0.0  |5  |∞  |        | SGD   |
2NN   |FedAVg|iid   | xxxx%            |xx    |100   |xxxhrs    | A       | 1.0  |5  |∞  |        | SGD   |
2NN   |FedAVg|iid   | xxxx%            |xx    |100   |xxxhrs    | A       | 0.5  |5  |∞  |        | SGD   |
2NN   |FedAVg|iid   | xxxx%            |xx    |100   |xxxhrs    | A       | 0.2  |5  |∞  |        | SGD   |
2NN   |FedAVg|iid   | xxxx%            |xx    |100   |xxxhrs    | A       | 0.1  |5  |∞  |        | SGD   |

##### 2NN/non-IID
Model |Method|Data  | Test acc (f,max) |R-98  |T Rnd |Time      | Machine | Frac | E | B | Lr     | Optim | Status
------|------|------| --------         |----- |----- |--------  |-------- | -----|---|---| -----  | ----- | ------
2NN   |FedAVg|N-iid | xxxx%            |xx    |100   |xxxhrs    | A       | 0.0  |5  |10 |        | SGD   |
2NN   |FedAVg|N-iid | xxxx%            |xx    |100   |xxxhrs    | A       | 1.0  |5  |10 |        | SGD   | 
2NN   |FedAVg|N-iid | xxxx%            |xx    |100   |xxxhrs    | A       | 0.5  |5  |10 |        | SGD   |
2NN   |FedAVg|N-iid | xxxx%            |xx    |100   |xxxhrs    | A       | 0.2  |5  |10 |        | SGD   |
2NN   |FedAVg|N-iid | xxxx%            |xx    |100   |xxxhrs    | A       | 0.1  |5  |10 |        | SGD   |
2NN   |FedAVg|N-iid | xxxx%            |xx    |100   |xxxhrs    | A       | 0.0  |5  |∞  |        | SGD   |
2NN   |FedAVg|N-iid | xxxx%            |xx    |100   |xxxhrs    | A       | 1.0  |5  |∞  |        | SGD   |
2NN   |FedAVg|N-iid | xxxx%            |xx    |100   |xxxhrs    | A       | 0.5  |5  |∞  |        | SGD   |
2NN   |FedAVg|N-iid | xxxx%            |xx    |100   |xxxhrs    | A       | 0.2  |5  |∞  |        | SGD   |
2NN   |FedAVg|N-iid | xxxx%            |xx    |100   |xxxhrs    | A       | 0.1  |5  |∞  |        | SGD   |



#### B. Experiment 2: increase local computation
* The fraction number is fixed at C=0.1
* The runs using optimized learning rate will be marked as "0.01/o"
* R-98 means the number of round where test acc hit 98%, similar for R-XX
* T Rnd means the total number of performed rounds
* Test acc (f,max) means final value and max value

##### CNN/IID
Model |Method|Data  | Test acc (f,max) |R-98  |T Rnd |Time      | Machine | Frac | E | B | Lr     | Optim | Status
------|------|------| --------         |----- |----  |--------  | -----   |---   |---| - | -----  | ----- | ------
CNN   |FedSGD|iid   | xxxxx%           |      |1000  |0.65hrs   | T       | 0.1  |1  |∞  | 0.15/o | SGD   | 
CNN   |FedAVg|iid   | 99.17%           |      |1000  |1.81hrs   | T       | 0.1  |5  |∞  | 0.08/o | SGD   | done ???
CNN   |FedAVg|iid   | 99.21%           |      |1000  |~0.76hrs  | T       | 0.1  |1  |50 | 0.2/o  | SGD   | done ???
CNN   |FedAVg|iid   | xxxxx%           |      |1000  |xxxxhrs   | T       | 0.1  |20 |∞  | 0.15/o | SGD   | 
CNN   |FedAVg|iid   | xxxxx%           |      |1000  |0.88hrs   | T       | 0.1  |1  |10 | 0.2/o  | SGD   | 
CNN   |FedAVg|iid   | %                |      |xxxx  |hrs       | T       | 0.1  |5  |50 | 0.2/o  | SGD   | 
CNN   |FedAVg|iid   | XXXXX%           |      |xxxx  |7.45hrs   | T       | 0.1  |20 |50 | 0.1/o  | SGD   | 
CNN   |FedAVg|iid   | %                |      |xxxx  |hrs       | T       | 0.1  |5  |10 | 0.2/o  | SGD   | 
CNN   |FedAVg|iid   | xxxxx%           |      |1000  |8.0hrs    | T       | 0.1  |20 |10 | 0.2/o  | SGD   | 
##### Remarks
1. For E=1, B=inf, after 1000 rounds, the test acc can still be improve much, the training loss can also be further reduced. It seems that either 1000 rounds is not enough or the learning rate needs optimized.
2. An make-do approach under the limited computational power, is to **use the same, non-optimized lr, with a lower target (e.g., 91%) for benchmarking the speed up for ever set of parameter combinations against the FedSGD**.
3. 400-500 rounds should be sufficient for E=5 B=50

##### CNN/non-IID
Model |Method|Data  | Test acc (f,max) |R-98  |T Rnd |Time      | Machine | Frac | E | B | Lr     | Optim | Status
------|------|------| --------         |----- |----  |--------  | -----   |---   |---| - | -----  | ----- | ------
CNN   |FedSGD|N-iid | %                |      |xxxx  |hrs       | T       | 0.1  |1  |∞  | 0.04/o | SGD   | 
CNN   |FedAVg|N-iid | %                |      |xxxx  |hrs       | T       | 0.1  |5  |∞  | 0.02/o | SGD   |
CNN   |FedAVg|N-iid | %                |      |xxxx  |hrs       | T       | 0.1  |1  |50 | 0.04/o | SGD   | 
CNN   |FedAVg|N-iid | %                |      |xxxx  |hrs       | T       | 0.1  |20 |∞  | 0.04/o | SGD   | 
CNN   |FedAVg|N-iid | %                |      |xxxx  |hrs       | T       | 0.1  |1  |10 | 0.04/o | SGD   | 
CNN   |FedAVg|N-iid | %                |      |xxxx  |hrs       | T       | 0.1  |5  |50 | 0.04/o | SGD   | 
CNN   |FedAVg|N-iid | %                |      |xxxx  |hrs       | T       | 0.1  |20 |50 | 0.04/o | SGD   | 
CNN   |FedAVg|N-iid | %                |      |xxxx  |hrs       | T       | 0.1  |5  |10 | 0.04/o | SGD   | 
CNN   |FedAVg|N-iid | %                |      |xxxx  |hrs       | T       | 0.1  |20 |10 | 0.04/o | SGD   | 

##### 2NN/IID
Model |Method|Data  | Test acc (f,max) |R-98  |T Rnd |Time      | Machine | Frac | E | B | Lr     | Optim | Status
------|------|------| --------         |----- |----  |--------  | -----   |---   |---| - | -----  | ----- | ------
2NN   |FedSGD|iid   | %                |      |xxxx  |hrs       | T       | 0.1  |1  |∞  | 0.04/o | SGD   | 
2NN   |FedAVg|iid   | %                |      |xxxx  |hrs       | T       | 0.1  |5  |∞  | 0.02/o | SGD   |
2NN   |FedAVg|iid   | %                |      |xxxx  |hrs       | T       | 0.1  |1  |50 | 0.04/o | SGD   | 
2NN   |FedAVg|iid   | %                |      |xxxx  |hrs       | T       | 0.1  |20 |∞  | 0.04/o | SGD   | 
2NN   |FedAVg|iid   | %                |      |xxxx  |hrs       | T       | 0.1  |1  |10 | 0.04/o | SGD   | 
2NN   |FedAVg|iid   | %                |      |xxxx  |hrs       | T       | 0.1  |5  |50 | 0.04/o | SGD   | 
2NN   |FedAVg|iid   | %                |      |xxxx  |hrs       | T       | 0.1  |20 |50 | 0.04/o | SGD   | 
2NN   |FedAVg|iid   | %                |      |xxxx  |hrs       | T       | 0.1  |5  |10 | 0.04/o | SGD   | 
2NN   |FedAVg|iid   | %                |      |xxxx  |hrs       | T       | 0.1  |20 |10 | 0.04/o | SGD   | 

##### 2NN/non-IID
Model |Method|Data  | Test acc (f,max) |R-98  |T Rnd |Time      | Machine | Frac | E | B | Lr     | Optim | Status
------|------|------| --------         |----- |----  |--------  | -----   |---   |---| - | -----  | ----- | ------
2NN   |FedSGD|N-iid | %                |      |xxxx  |hrs       | T       | 0.1  |1  |∞  | 0.04/o | SGD   | 
2NN   |FedAVg|N-iid | %                |      |xxxx  |hrs       | T       | 0.1  |5  |∞  | 0.02/o | SGD   |
2NN   |FedAVg|N-iid | %                |      |xxxx  |hrs       | T       | 0.1  |1  |50 | 0.04/o | SGD   | 
2NN   |FedAVg|N-iid | %                |      |xxxx  |hrs       | T       | 0.1  |20 |∞  | 0.04/o | SGD   | 
2NN   |FedAVg|N-iid | %                |      |xxxx  |hrs       | T       | 0.1  |1  |10 | 0.04/o | SGD   | 
2NN   |FedAVg|N-iid | %                |      |xxxx  |hrs       | T       | 0.1  |5  |50 | 0.04/o | SGD   | 
2NN   |FedAVg|N-iid | %                |      |xxxx  |hrs       | T       | 0.1  |20 |50 | 0.04/o | SGD   | 
2NN   |FedAVg|N-iid | %                |      |xxxx  |hrs       | T       | 0.1  |5  |10 | 0.04/o | SGD   | 
2NN   |FedAVg|N-iid | %                |      |xxxx  |hrs       | T       | 0.1  |20 |10 | 0.04/o | SGD   | 


#### C. Experiment 3: CIFAR10 learning performance

##### Fixed federated setting VS FedSGD and SGD
* Fraction of users is fixed at C=0.1
* FedSGD and FedAVg use fixed E=5, and FedAvg use fixed B=50
* Decay means the learning-rate decay
* Since E, B are fixed then total number of mini-batch updates $n=R\times250$

Model |Method|Data  | Test acc (f,max) |R-98  |T Rnd |Time      | Machine | Frac | E | B | Lr     |Decay  | Optim | Status
------|------|------| --------         |----- |----  |--------  | -----   |---   |---| - | -----  |------ | ----- | ------
CNN   |SGD   |iid   | %                |      |xxxx  |hrs       | T       |      |   |50 |        |       | SGD   | 
CNN   |FedSGD|iid   | %                |      |xxxx  |hrs       | T       | 0.1  |5  |∞  |        |       | SGD   |
CNN   |FedAVg|iid   | %                |      |xxxx  |hrs       | T       | 0.1  |5  |50 |        |       | SGD   | 
CNN   |FedAVg|iid   | %                |      |xxxx  |hrs       | T       | 0.1  |5  |50 |        |       | SGD   | 
CNN   |FedAVg|iid   | %                |      |xxxx  |hrs       | T       | 0.1  |5  |50 |        |       | SGD   | 
CNN   |FedAVg|iid   | %                |      |xxxx  |hrs       | T       | 0.1  |5  |50 |        |       | SGD   | 
CNN   |FedAVg|iid   | %                |      |xxxx  |hrs       | T       | 0.1  |5  |50 |        |       | SGD   | 
CNN   |FedAVg|iid   | %                |      |xxxx  |hrs       | T       | 0.1  |5  |50 |        |       | SGD   | 

##### Varying federated setting VS SGD
* The learning rates of every setting are seem to be fixed at the vanilla FL paper
* Each setting runs same number of mini-batch updates $n=?$
* Totoal number of rounds of each setting is $R=n/(E\timesB)$

Model |Method|Data  | Test acc (f,max) |R-98  |T Rnd |Time      | Machine | Frac | E | B | Lr     | Optim | Status
------|------|------| --------         |----- |----  |--------  | -----   |---   |---| - | -----  | ----- | ------
CNN   |SGD   |iid   | %                |      |xxxx  |hrs       | T       |      |   |50 |        | SGD   | 
CNN   |FedAVg|iid   | %                |      |xxxx  |hrs       | T       | 0.0  |1  |50 |        | SGD   | 
CNN   |FedAVg|iid   | %                |      |xxxx  |hrs       | T       | 0.1  |10 |50 |        | SGD   | 
CNN   |FedAVg|iid   | %                |      |xxxx  |hrs       | T       | 0.1  |20 |50 |        | SGD   | 
CNN   |FedAVg|iid   | %                |      |xxxx  |hrs       | T       | 1.0  |5  |50 |        | SGD   | 
CNN   |FedAVg|iid   | %                |      |xxxx  |hrs       | T       | 0.1  |5  |50 |        | SGD   | 
CNN   |FedAVg|iid   | %                |      |xxxx  |hrs       | T       | 0.1  |5  |50 |        | SGD   | 
#### Appendix: Training time summary
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
