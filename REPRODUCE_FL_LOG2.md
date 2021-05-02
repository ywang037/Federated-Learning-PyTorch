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
##### Generic remarks
1. Some test runs are unstable under searched learning rate, this is
    * either because the learning rate is too close to instability region
    * or due to the limited memory and computational power of the machine 

#### A. Experiment 1: increase parallism
* The runs using optimized learning rate will be marked as "0.01/o"
* Set 96%, 98% as targets for CNN, 2NN is to avoid the needed rounds from being too large to complete in time.
* R-98 means the number of round where test acc hit 98%, similar for R-XX
* T Rnd means the total number of performed rounds
* Test acc (f,max) means final value and max value 
* The time taken for C=1.0 is formidable, even for IID cases. So, experiment for C=1.0 is cancelled for now.
    * one could consider 200 rounds for C=1.0, E=5, B=10 of non-IID, in order to complete in allowed timeline.

##### CNN/IID
Model |Method|Data  | Test acc (f,max) |R-98     |T Rnd |Time      | Machine | Frac | E | B | Lr     | Optim | Status
------|------|------| --------         |-----    |----- |--------  |-------- | -----|---|---| -----  | ----- | ------
CNN   |FedAVg|iid   | 98.58%,99.23%    |20       |500   |0.42hrs   | A       | 0.0  |5  |10 | 0.1/o  | SGD   | done
CNN   |FedAVg|iid   | 99.36%,99.36%    |6  (3.3x)|100   |0.38hrs   | A       | 0.1  |5  |10 | 0.1/o  | SGD   | done
CNN   |FedAVg|iid   | 99.35%,99.36%    |5  (4.0x)|100   |0.69hrs   | A       | 0.2  |5  |10 | 0.1/o  | SGD   | done
CNN   |FedAVg|iid   | 99.29%,99.32%    |5  (4.0x)|100   |1.54hrs   | A       | 0.5  |5  |10 | 0.1/o  | SGD   | done
CNN   |FedAVg|iid   |                  |         |      |          | A       | 1.0  |5  |10 |        | SGD   | cancelled
CNN   |FedAVg|iid   | 98.19%,98.45%    |88       |100   |0.05hrs   | A       | 0.0  |5  |∞  | 0.1/o  | SGD   | done
CNN   |FedAVg|iid   | 98.36%,98.37%    |67 (1.3x)|100   |0.18hrs   | T       | 0.1  |5  |∞  | 0.2/o  | SGD   | done
CNN   |FedAVg|iid   | 98.66%,98.70%    |54 (1.6x)|100   |0.32hrs   | T       | 0.2  |5  |∞  | 0.2/o  | SGD   | done
CNN   |FedAVg|iid   | 98.48%,98.52%    |55 (1.6x)|100   |0.78hrs   | A       | 0.5  |5  |∞  | 0.2/o  | SGD   | done
CNN   |FedAVg|iid   |                  |         |100   |          | A       | 1.0  |5  |∞  |        | SGD   | cancelled

##### Remarks
1. {E=5, B=∞} cannot reach 99% in allowed time i.e., within 100 rounds.
2. As the fraction paramter C increases from 0.1 to 0.5, there is no significant speed up improvemnt observed, which is in line with the experimental results reported in the vanilla FL paper (referred to as vanilla FL results for short).


##### CNN/non-IID
Model |Method|Data  | Test acc (f,max) |R-98      |T Rnd |Time      | Machine | Frac | E | B | Lr     | Optim | Status
------|------|------| --------         |-----     |----- |--------  |-------- | -----|---|---| -----  | ----- | ------
CNN   |FedAVg|N-iid | 99.27%,99.38%    |416       |1500  |1.4hrs    | A       | 0.0  |5  |10 | 0.04/o | SGD   | done
CNN   |FedAVg|N-iid | 99.21%,99.30%    |75  (5.5x)|1500  |6.25hrs   | A       | 0.1  |5  |10 | 0.04/o | SGD   | done
CNN   |FedAVg|N-iid | 99.11%,99.19%    |40 (10.4x)|250   |1.67hrs   | A       | 0.2  |5  |10 | 0.04/o | SGD   | done
CNN   |FedAVg|N-iid | 98.95%,99.07%    |42  (9.9x)|200   |3.97hrs   | A       | 0.5  |5  |10 | 0.04/o | SGD   | done
CNN   |FedAVg|N-iid |                  |          |      |          | A       | 1.0  |5  |10 |        | SGD   | cancelled
CNN   |FedAVg|N-iid | 98.62%,98.98%    |778       |1500  |1.00hrs   | A       | 0.0  |5  |∞  | 0.02/o | SGD   | done
CNN   |FedAVg|N-iid | 98.92%,99.07%    |323 (2.4x)|1500  |3.08hrs   | A       | 0.1  |5  |∞  | 0.04/o | SGD   | done
CNN   |FedAVg|N-iid | 98.99%,99.08%    |319 (2.4x)|1500  |5.54rs    | A       | 0.2  |5  |∞  | 0.04/o | SGD   | done
CNN   |FedAVg|N-iid | 98.36%,98.36%    |318 (2.5x)|400   |3.40hrs   | A       | 0.5  |5  |∞  | 0.04/o | SGD   | done
CNN   |FedAVg|N-iid |                  |          |      |          | A       | 1.0  |5  |∞  |        | SGD   | cancelled

##### Remarks
1. Similar to IID cases, {E=5, B=∞} cannot reach 99% in allowed time, i.e., within 400 rounds.
2. Similar to IID cases, increasing the fraction paramter C from 0.1 to 0.5 does not necessarily improve or improves little the speed up, which is in line with the vanilla FL results.


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
* The fraction number is fixed at C=0.1.
* The runs using optimized learning rate will be marked as "0.01/o".
* R-98 means the number of round where test acc hit 98%, similar for R-XX.
* T Rnd means the total number of performed rounds.
* Test acc (f,max) means final value and max value.
* Runs with status marked as done/d means discarded results.

##### CNN/IID
Model |Method|Data  | Test acc (f,max) |R-98  |R-99  |T Rnd |Time      | Machine | Frac | E | B | Lr     | Optim | Status
------|------|------| --------         |----- |----- |----  |--------  | -----   |---   |---| - | -----  | ----- | ------
CNN   |FedSGD|iid   | 98.94%,99.01%    | 200  | 573  |600   |0.40hrs   | T       | 0.1  |1  |∞  | 0.2/o  | SGD   | done/d
CNN   |FedSGD|iid   | 16.34%,99.08%    | 210  | 603  |1000  |0.66hrs   | T       | 0.1  |1  |∞  | 0.15/o | SGD   | done/d
CNN   |FedSGD|iid   | %,%              |      |      |1000  |hrs       | T       | 0.1  |1  |∞  | 0.1/o  | SGD   | ???
CNN   |FedAVg|iid   | 99.02%,99.09%    | 57   | 253  |600   |1.10hrs   | T       | 0.1  |5  |∞  | 0.1/o  | SGD   | done 
CNN   |FedAVg|iid   | 99.26%,99.41%    | 19   | 55   |600   |0.41hrs   | T       | 0.1  |1  |50 | 0.2/o  | SGD   | done
CNN   |FedAVg|iid   | xxxxx%           |      |      |600   |xxxxhrs   | T       | 0.1  |20 |∞  | 0.15/o | SGD   | to run on T
CNN   |FedAVg|iid   | xxxxx%           |      |      |600   |0.88hrs   | T       | 0.1  |1  |10 | 0.15/o | SGD   | to run on T
CNN   |FedAVg|iid   | %                |      |      |600   |hrs       | T       | 0.1  |5  |50 | 0.2/o  | SGD   | to run on T
CNN   |FedAVg|iid   | XXXXX%           |      |      |600   |7.45hrs   | T       | 0.1  |20 |50 | 0.1/o  | SGD   | 
CNN   |FedAVg|iid   | %                |      |      |600   |hrs       | T       | 0.1  |5  |10 | 0.2/o  | SGD   | 
CNN   |FedAVg|iid   | xxxxx%           |      |      |600   |8.0hrs    | T       | 0.1  |20 |10 | 0.2/o  | SGD   | 

##### Remarks
1. For {E=1, B=inf}, lr=0.2 and lr=0.15 produce similar results in 600 rounds, but lr=0.15 become unstable at last few rounds near 1000, so would the larger lr=0.2. It seems that lr=0.1 might be more reasonable learning rate.
2. For {E=1, B=10}, lr=0.2 is unstable in real test run, try {0.15, 0.1, 0.07} instead. The other parameter combinations can follow similar approach for dealing instability under large lr.
3. It can observed that the more local computation, the quicker the convergence (or the higher test acc that can be achived within the identical number or rounds)
4. Larger learning rates in general lead to quicker convergence or higher test acc in a given number of rounds, at the expense of being prone to instability. 


##### CNN/non-IID
Model |Method|Data  | Test acc (f,max) |R-98  |R-99  |T Rnd |Time      | Machine | Frac | E | B | Lr     | Optim | Status
------|------|------| --------         |----- |      |----  |--------  | -----   |---   |---| - | -----  | ----- | ------
CNN   |FedSGD|N-iid | %                |      |      |1500  |hrs       | T       | 0.1  |1  |∞  | 0.04/o | SGD   | run on A
CNN   |FedAVg|N-iid | 98.73%,98.81%    | 264  |      |600   |1.25hrs   | A       | 0.1  |5  |∞  | 0.04/o | SGD   | done
CNN   |FedAVg|N-iid | 97.65%,98.15%    | 567  |      |600   |1.17hrs   | A       | 0.1  |5  |∞  | 0.02/o | SGD   | done/d
CNN   |FedAVg|N-iid | 98.54%,98.97%    | 152  |      |600   |0.65hrs   | A       | 0.1  |1  |50 | 0.04/o | SGD   | done
CNN   |FedAVg|N-iid |                  |      |      |600   |hrs       | A       | 0.1  |20 |∞  | 0.04/o | SGD   | run on A
CNN   |FedAVg|N-iid | 99.17%,99.35%    | 79   | 204  |600   |0.70hrs   | T       | 0.1  |1  |10 | 0.04/o | SGD   | done
CNN   |FedAVg|N-iid | 98.91%,99.06%    | 109  | 344  |600   |1.61hrs   | A       | 0.1  |5  |50 | 0.04/o | SGD   | done
CNN   |FedAVg|N-iid | 99.04%,99.16%    | 73   | 351  |600   |4.12hrs   | T       | 0.1  |20 |50 | 0.04/o | SGD   | done
CNN   |FedAVg|N-iid | 99.22%,99.30%    | 42   | 145  |600   |2.58hrs   | A       | 0.1  |5  |10 | 0.04/o | SGD   | done
CNN   |FedAVg|N-iid | 99.15%,99.23%    | 55   | 175  |600   |7.36hrs   | T       | 0.1  |20 |10 | 0.04/o | SGD   | done

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
