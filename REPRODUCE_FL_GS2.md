# Work logs and notes on reproducing vanilla FL paper - Part C-2: grid search of learning rate

### Approach (fundamental principle)
* WY's MLP and CNN models is used instead of original models found in AshwinRJ's repository
* The grid search conducted below follows the approach described in Part A section III:
    1. Coarse search using a bigger resolution, a factor of 10 or 0.1, say, start from 1e-5, then 1e-4, 1e-3, 1e-2, 1e-1, and 1.0,
    2. Finer search using a smaller resolution, factor-2, i.e., {2, 4, 8}, between two best values found in the previous coarse search.

### Approach for IID data
* Initial coarse searches in a grid {1e-4, 1e-3, 1e-2, 1e-1, 1.0}, best values were found around 1e-2 and 1e-1.
* It is found that, for WY's CNN model of learning MNIST, best learning rate are most likely to appear around {0.08, 0.1, 0.16, 0.2}. Therefore, one can reduce the search range to {0.08, 0.1, 0.2} firstly, since these three values are most likely to be the best learning rate. 0.16 can be checked additionaly.
* Most of the completed search show that WY's CNN model of learning MNIST leads to monototically increased test accuracy. Therefore, one may use 100 rounds instead of 200 rounds for searching the learning rate.

### Approach for Non-IID data
* Initial coarse searches in a grid {1e-5, 1e-4, 1e-3, 1e-2, 1e-1}, best values were found around 1e-3 and 1e-2
* It is found that, for WY's CNN model of learning MNIST, best learning rate are most likely to appear around {0.008, 0.01, 0.02, 0.04}. Therefore, one can reduce the search range to {0.01, 0.02, 0.04} firstly, since these three values are most likely to be the best learning rate. 0.008, 0.06 can be checked additionaly.
* It was observed that 0.02 are most likely to be the best learning rate for non-IID cases.

### I. Baseline CIFAR10 learning with *torch cnn* and *tf cnn* model 
#### Torch cnn only (27 April 2021)
N/A

#### Torch cnn and tf cnn (28 April 2021)
N/A


### II. Baseline MNIST learning with *2NN* and *CNN* models
Model | Test acc | Time     | Batch size | Epochs | Lr/O   | Optim
------| -------- | -------- | ---------- | ------ | ------ | ---------
2NN   | 97.12%   | 1765s    | 100        | 200    | 0.01?  | vanilla SGD 
CNN   | 99.09%   | 2014s    | 100        | 200    | 0.01?  | vanilla SGD 

Both trianed models might be used for warm start in future training.

### III. FedAvg MNIST learning with *2NN* and *CNN* models 
#### A. Experiment 1: increase parallism
* Lr/O means the optimized value of learning rate
* The learning rate found using coarse search will be marked as "done cs"
* The learning rate found using further finer search will be marked as "done fs"



##### CNN/IID
* Default number of rounds is 200, if any run differs, then it will be marked as XX.XX%-XXX
* If the test acc at final round differs little, then check the final 10 rounds and choose the one with higher average value.
* If two lr ties in test acc, 
    1. 0.08 ties with 0.1, then take 0.1, to use larger ones for quicker convergence hopefully 
    2. 0.1 ties with 0.2, then take 0.1, to use smaller ones for better stability hopeflly 
    3. 0.08 ties with 0.2, then take 0.08, to use smaller ones for better stability hopeflly 

Model |Method|Data  | Val test acc |Time used | Machine | Frac | E | B | Lr/O  | Optim | Status
------|------|------| --------     |--------- | --------| -----|---|---| ----- | ----- | ------
CNN   |FedAVg|iid   | 98.41%       |0.09hrs   | A       | 0.0  |5  |10 | 0.1   | SGD   | fs done
CNN   |FedAVg|iid   | %            |3.6hrs    | A       | 1.0  |5  |10 |       | SGD   | cancelled
CNN   |FedAVg|iid   | 98.49%       |0.56hrs   | A       | 0.5  |5  |10 | 0.08  | SGD   | fs done
CNN   |FedAVg|iid   | 98.50%       |0.27hrs   | A       | 0.2  |5  |10 | 0.2   | SGD   | fs done
CNN   |FedAVg|iid   | 98.43%       |0.17hrs   | A       | 0.1  |5  |10 | 0.08  | SGD   | fs done
CNN   |FedAVg|iid   | 98.36%       |0.09hrs   | A       | 0.0  |5  |∞  | 0.1   | SGD   | fs done
CNN   |FedAVg|iid   | %            |xxxhrs    | A       | 1.0  |5  |∞  |       | SGD   | cancelled
CNN   |FedAVg|iid   | 98.46%       |0.95hrs   | A       | 0.5  |5  |∞  | 0.1   | SGD   | fs done
CNN   |FedAVg|iid   | 98.61%       |0.27hrs   | A       | 0.2  |5  |∞  | 0.2   | SGD   | fs done
CNN   |FedAVg|iid   | 98.45%       |0.18hrs   | A       | 0.1  |5  |∞  | 0.2   | SGD   | fs done

##### Remarks
N/A

##### CNN/non-IID
* Default number of rounds is 400, if any run differs, then it will be marked as XX.XX%-XXX
* If the test acc at final round differs little, then check the final 20 rounds and choose the one with higher average value.
* If two lr ties in test acc, 
    1. 0.01 ties with 0.02, then take 0.02, since the validation loss of 0.02 decreases more smoothly.
    2. 0.02 ties with 0.04, then take 0.02, since 0.04 are closer to instability zone. 
    3. 0.01 ties with 0.04, then take 0.04, since the validation loss of 0.02 decreases more smoothly.
* Watch out the "FAKE NEWS": in {0.02, 0.04, 0.06}, larger lr may give better final stage test acc, but is also more likely to present instability. If the difference between test acc is not significant, then it could be safer to use smaller ones, e.g., 0.02 over 0.04

Model |Method|Data  | Val test acc |Time used | Machine | Frac | E | B | Lr/O  | Optim | Status
------|------|------| --------     |--------- | --------| -----|---|---| ----- | ----- | ------
CNN   |FedAVg|N-iid | 95.44%       |0.18hrs   | A       | 0.0  |5  |10 | 0.02  | SGD   | fs done
CNN   |FedAVg|N-iid | xxxx%        |xxxhrs    | A       | 1.0  |5  |10 | 0.01  | SGD   | cancelled
CNN   |FedAVg|N-iid | xxxx%        |xxxhrs    | A       | 0.5  |5  |10 | 0.01  | SGD   | 
CNN   |FedAVg|N-iid | 94.69%       |0.55hrs   | A       | 0.2  |5  |10 | 0.04  | SGD   | fs done
CNN   |FedAVg|N-iid | 96.90%       |0.35hrs   | A       | 0.1  |5  |10 | 0.04  | SGD   | fs done
CNN   |FedAVg|N-iid | xxxx%        |xxxhrs    | A       | 0.0  |5  |∞  | 0.01  | SGD   | 2B run on A
CNN   |FedAVg|N-iid | xxxx%        |xxxhrs    | A       | 1.0  |5  |∞  | 0.01  | SGD   | cancelled
CNN   |FedAVg|N-iid | xxxx%        |xxxhrs    | A       | 0.5  |5  |∞  | 0.01  | SGD   |
CNN   |FedAVg|N-iid | xxxx%        |xxxhrs    | A       | 0.2  |5  |∞  | 0.01  | SGD   | 2B run on A
CNN   |FedAVg|N-iid | xxxx%        |xxxhrs    | A       | 0.1  |5  |∞  | 0.01  | SGD   | 2B run on A

##### 2NN/IID
Model |Method|Data  | Val test acc |Time used | Machine | Frac | E | B | Lr/O  | Optim | Status
------|------|------| --------     |--------- | --------| -----|---|---| ----- | ----- | ------
2NN   |FedAVg|iid   | xxxx%        |xxxhrs    | A       | 0.0  |5  |10 | 0.01  | SGD   |
2NN   |FedAVg|iid   | xxxx%        |xxxhrs    | A       | 1.0  |5  |10 | 0.01  | SGD   | cancelled
2NN   |FedAVg|iid   | xxxx%        |xxxhrs    | A       | 0.5  |5  |10 | 0.01  | SGD   | 
2NN   |FedAVg|iid   | xxxx%        |xxxhrs    | A       | 0.2  |5  |10 | 0.01  | SGD   | 
2NN   |FedAVg|iid   | xxxx%        |xxxhrs    | A       | 0.1  |5  |10 | 0.01  | SGD   |
2NN   |FedAVg|iid   | xxxx%        |xxxhrs    | A       | 0.0  |5  |∞  | 0.01  | SGD   |
2NN   |FedAVg|iid   | xxxx%        |xxxhrs    | A       | 1.0  |5  |∞  | 0.01  | SGD   | cancelled
2NN   |FedAVg|iid   | xxxx%        |xxxhrs    | A       | 0.5  |5  |∞  | 0.01  | SGD   |
2NN   |FedAVg|iid   | xxxx%        |xxxhrs    | A       | 0.2  |5  |∞  | 0.01  | SGD   |
2NN   |FedAVg|iid   | xxxx%        |xxxhrs    | A       | 0.1  |5  |∞  | 0.01  | SGD   |

##### 2NN/non-IID
Model |Method|Data  | Val test acc |Time used | Machine | Frac | E | B | Lr/O  | Optim | Status
------|------|------| --------     |--------- | --------| -----|---|---| ----- | ----- | ------
2NN   |FedAVg|N-iid | xxxx%        |xxxhrs    | A       | 0.0  |5  |10 | 0.01  | SGD   |
2NN   |FedAVg|N-iid | xxxx%        |xxxhrs    | A       | 1.0  |5  |10 | 0.01  | SGD   | cancelled
2NN   |FedAVg|N-iid | xxxx%        |xxxhrs    | A       | 0.5  |5  |10 | 0.01  | SGD   | 
2NN   |FedAVg|N-iid | xxxx%        |xxxhrs    | A       | 0.2  |5  |10 | 0.01  | SGD   | 
2NN   |FedAVg|N-iid | xxxx%        |xxxhrs    | A       | 0.1  |5  |10 | 0.01  | SGD   |
2NN   |FedAVg|N-iid | xxxx%        |xxxhrs    | A       | 0.0  |5  |∞  | 0.01  | SGD   |
2NN   |FedAVg|N-iid | xxxx%        |xxxhrs    | A       | 1.0  |5  |∞  | 0.01  | SGD   | cancelled
2NN   |FedAVg|N-iid | xxxx%        |xxxhrs    | A       | 0.5  |5  |∞  | 0.01  | SGD   |
2NN   |FedAVg|N-iid | xxxx%        |xxxhrs    | A       | 0.2  |5  |∞  | 0.01  | SGD   |
2NN   |FedAVg|N-iid | xxxx%        |xxxhrs    | A       | 0.1  |5  |∞  | 0.01  | SGD   |



#### B. Experiment 2: increase local computation
* The fraction number is fixed at C=0.1
* The learning rate found using coarse search will be marked as "done cs"
* The learning rate found using further finer search will be marked as "done fs"

##### Remarks
N/A

##### CNN/IID
* Default number of rounds is 200, if any run differs, then it will be marked as XX.XX%-XXX

Model |Method|Data  | Val test acc|Time used | Machine | Frac | E | B | Lr/O     | Optim | Status
------|------|------| --------    |--------  |-------- | -----|---|---| -----    | ------| -----
CNN   |FedSGD|iid   | 98.54%      |5.5mins   | T       | 0.1  |1  |∞  | 0.2      | SGD   | fs done
CNN   |FedAVg|iid   | 98.68%      |10.3mins  | T       | 0.1  |5  |∞  | 0.08     | SGD   | fs done
CNN   |FedAVg|iid   | 98.66%      |5.8mins   | T       | 0.1  |1  |50 | 0.2      | SGD   | fs done
CNN   |FedAVg|iid   | 98.51%      |28.2mins  | T       | 0.1  |20 |∞  | 0.16/0.2 | SGD   | fs done
CNN   |FedAVg|iid   | 98.51%      |5.8mins   | T       | 0.1  |1  |10 | 0.2      | SGD   | fs done
CNN   |FedAVg|iid   | 98.48%      |10.5mins  | T       | 0.1  |5  |50 | 0.2      | SGD   | fs done
CNN   |FedAVg|iid   | 98.61%      |28.1mins  | T       | 0.1  |20 |50 | 0.1      | SGD   | fs done
CNN   |FedAVg|iid   | 98.50%      |10.5mins  | T       | 0.1  |5  |10 | 0.2      | SGD   | fs done
CNN   |FedAVg|iid   | 98.55%      |46.7mins  | T       | 0.1  |20 |10 | 0.2?     | SGD   | fs done

##### CNN/non-IID
Model |Method|Data  | Val test acc|Time used | Machine | Frac | E | B | Lr/O  | Optim | Status
------|------|------| --------    |--------  |-------- | -----|---|---| ----- | ------| -----
CNN   |FedSGD|N-iid | %           |hrs       | T       | 0.1  |1  |∞  | 0.01  | SGD   | run on T
CNN   |FedAVg|N-iid | %           |hrs       | T       | 0.1  |5  |∞  | 0.01  | SGD   | 
CNN   |FedAVg|N-iid | %           |hrs       | T       | 0.1  |1  |50 | 0.01  | SGD   | 
CNN   |FedAVg|N-iid | %           |hrs       | T       | 0.1  |20 |∞  | 0.01  | SGD   | 
CNN   |FedAVg|N-iid | %           |hrs       | T       | 0.1  |1  |10 | 0.01  | SGD   | 
CNN   |FedAVg|N-iid | %           |hrs       | T       | 0.1  |5  |50 | 0.01  | SGD   | 
CNN   |FedAVg|N-iid | %           |hrs       | T       | 0.1  |20 |50 | 0.01  | SGD   | 
CNN   |FedAVg|N-iid | %           |hrs       | T       | 0.1  |5  |10 | 0.01  | SGD   | 
CNN   |FedAVg|N-iid | %           |hrs       | T       | 0.1  |20 |10 | 0.01  | SGD   | 

##### 2NN/IID
Model |Method|Data  | Val test acc|Time used | Machine | Frac | E | B | Lr/O  | Optim | Status
------|------|------| --------    |--------  |-------- | -----|---|---| ----- | ------| -----
2NN   |FedSGD|iid   | %           |hrs       | T       | 0.1  |1  |∞  | 0.01  | SGD   | 
2NN   |FedAVg|iid   | %           |hrs       | T       | 0.1  |5  |∞  | 0.01  | SGD   | 
2NN   |FedAVg|iid   | %           |hrs       | T       | 0.1  |1  |50 | 0.01  | SGD   | 
2NN   |FedAVg|iid   | %           |hrs       | T       | 0.1  |20 |∞  | 0.01  | SGD   | 
2NN   |FedAVg|iid   | %           |hrs       | T       | 0.1  |1  |10 | 0.01  | SGD   | 
2NN   |FedAVg|iid   | %           |hrs       | T       | 0.1  |5  |50 | 0.01  | SGD   | 
2NN   |FedAVg|iid   | %           |hrs       | T       | 0.1  |20 |50 | 0.01  | SGD   | 
2NN   |FedAVg|iid   | %           |hrs       | T       | 0.1  |5  |10 | 0.01  | SGD   | 
2NN   |FedAVg|iid   | %           |hrs       | T       | 0.1  |20 |10 | 0.01  | SGD   | 


##### 2NN/non-IID
Model |Method|Data  | Val test acc|Time used | Machine | Frac | E | B | Lr/O  | Optim | Status
------|------|------| --------    |--------  |-------- | -----|---|---| ----- | ------| -----
2NN   |FedSGD|N-iid | %           |hrs       | T       | 0.1  |1  |∞  | 0.01  | SGD   | 
2NN   |FedAVg|N-iid | %           |hrs       | T       | 0.1  |5  |∞  | 0.01  | SGD   | 
2NN   |FedAVg|N-iid | %           |hrs       | T       | 0.1  |1  |50 | 0.01  | SGD   | 
2NN   |FedAVg|N-iid | %           |hrs       | T       | 0.1  |20 |∞  | 0.01  | SGD   | 
2NN   |FedAVg|N-iid | %           |hrs       | T       | 0.1  |1  |10 | 0.01  | SGD   | 
2NN   |FedAVg|N-iid | %           |hrs       | T       | 0.1  |5  |50 | 0.01  | SGD   | 
2NN   |FedAVg|N-iid | %           |hrs       | T       | 0.1  |20 |50 | 0.01  | SGD   | 
2NN   |FedAVg|N-iid | %           |hrs       | T       | 0.1  |5  |10 | 0.01  | SGD   | 
2NN   |FedAVg|N-iid | %           |hrs       | T       | 0.1  |20 |10 | 0.01  | SGD   | 

#### C. Training time summary
Model |Method| Data |Time/rnd | 100-rnd time    | Machine |Frac | E | B | Lr    | Optim
------|------|------|-------- | --------------  |-------- |-----|---|---| ----- | ---------
CNN   |FedSGD|M-iid | 2.2s    | 3.7mins/0.06hrs | T       |0.1  |1  | ∞ | 0.01  | SGD
CNN   |FedAVg|M-iid | 5.6s    | 9.3mins/0.16hrs | T       |0.1  |5  | ∞ | 0.01  | SGD
CNN   |FedAvg|M-iid | 129.12s | 3.6hrs          | A       |1.0  |5  |10 | 0.01  | SGD 
CNN   |FedAvg|M-iid | 39.4s   | 1.1hrs          | A       |0.1  |20 |10 | 0.01  | SGD
CNN   |FedAvg|M-iid | 25.2s   | 42mins/0.7hrs   | A       |0.2  |5  |10 | 0.01  | SGD
CNN   |FedAvg|M-iid | 42.34s  | 1.2hrs          | T       |0.1  |20 |10 | 0.01  | SGD
CNN   |FedAvg|M-iid | 2.2s    | 3.6mins/0.06hrs | T       |0.1  |1  |50 | 0.01  | SGD
