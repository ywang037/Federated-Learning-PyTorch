# Work logs and notes on reproducing vanilla FL paper - Part C-2: grid search of learning rate

### Approach (fundamental principle)
* WY's MLP and CNN models is used instead of original models found in AshwinRJ's repository
* The grid search conducted below follows the approach described in Part A section III:
    1. Coarse search using a bigger resolution, a factor of 10 or 0.1, say, start from 1e-5, then 1e-4, 1e-3, 1e-2, 1e-1, and 1.0,
    2. Finer search using a smaller resolution, factor-2, i.e., {2, 4, 8}, between two best values found in the previous coarse search.

### Approach (update)
* It is found that, for WY's CNN model of learning MNIST, best learning rate are most likely to appera around {0.08, 0.1, 0.15, 0.2}. Therefore, one can reduce the search range to {0.04, 0.08, 0.1, 0.16, 0.2}
* Most of the completed search show that WY's CNN model of learning MNIST leads to monototically increased test accuracy. Therefore, one may use 100 rounds instead of 200 rounds for searching the learning rate.

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

Model |Method|Data  | Val test acc |Time used | Machine | Frac | E | B | Lr/O  | Optim | Status
------|------|------| --------     |--------- | --------| -----|---|---| ----- | ----- | ------
CNN   |FedAVg|iid   | xxxx%        |xxxhrs    | A       | 0.0  |5  |10 | 0.01  | SGD   |
CNN   |FedAVg|iid   | xxxx%        |3.6hrs    | A       | 1.0  |5  |10 | 0.01  | SGD   | 
CNN   |FedAVg|iid   | xxxx%        |1.7hrs    | A       | 0.5  |5  |10 | 0.01  | SGD   | 
CNN   |FedAVg|iid   | xxxx%        |0.7hrs    | A       | 0.2  |5  |10 | 0.01  | SGD   | 
CNN   |FedAVg|iid   | xxxx%        |xxxhrs    | A       | 0.1  |5  |10 | 0.01  | SGD   |
CNN   |FedAVg|iid   | xxxx%        |xxxhrs    | A       | 0.0  |5  |∞  | 0.01  | SGD   |
CNN   |FedAVg|iid   | xxxx%        |xxxhrs    | A       | 1.0  |5  |∞  | 0.01  | SGD   |
CNN   |FedAVg|iid   | xxxx%        |xxxhrs    | A       | 0.5  |5  |∞  | 0.01  | SGD   |
CNN   |FedAVg|iid   | xxxx%        |xxxhrs    | A       | 0.2  |5  |∞  | 0.01  | SGD   |
CNN   |FedAVg|iid   | xxxx%        |xxxhrs    | A       | 0.1  |5  |∞  | 0.01  | SGD   |

##### Remarks
N/A

##### CNN/non-IID
Model |Method|Data  | Val test acc |Time used | Machine | Frac | E | B | Lr/O  | Optim | Status
------|------|------| --------     |--------- | --------| -----|---|---| ----- | ----- | ------
CNN   |FedAVg|N-iid | xxxx%        |xxxhrs    | A       | 0.0  |5  |10 | 0.01  | SGD   |
CNN   |FedAVg|N-iid | xxxx%        |3.6hrs    | A       | 1.0  |5  |10 | 0.01  | SGD   | 
CNN   |FedAVg|N-iid | xxxx%        |1.7hrs    | A       | 0.5  |5  |10 | 0.01  | SGD   | 
CNN   |FedAVg|N-iid | xxxx%        |0.7hrs    | A       | 0.2  |5  |10 | 0.01  | SGD   | 
CNN   |FedAVg|N-iid | xxxx%        |xxxhrs    | A       | 0.1  |5  |10 | 0.01  | SGD   |
CNN   |FedAVg|N-iid | xxxx%        |xxxhrs    | A       | 0.0  |5  |∞  | 0.01  | SGD   |
CNN   |FedAVg|N-iid | xxxx%        |xxxhrs    | A       | 1.0  |5  |∞  | 0.01  | SGD   |
CNN   |FedAVg|N-iid | xxxx%        |xxxhrs    | A       | 0.5  |5  |∞  | 0.01  | SGD   |
CNN   |FedAVg|N-iid | xxxx%        |xxxhrs    | A       | 0.2  |5  |∞  | 0.01  | SGD   |
CNN   |FedAVg|N-iid | xxxx%        |xxxhrs    | A       | 0.1  |5  |∞  | 0.01  | SGD   |

##### 2NN/IID
Model |Method|Data  | Val test acc |Time used | Machine | Frac | E | B | Lr/O  | Optim | Status
------|------|------| --------     |--------- | --------| -----|---|---| ----- | ----- | ------
2NN   |FedAVg|iid   | xxxx%        |xxxhrs    | A       | 0.0  |5  |10 | 0.01  | SGD   |
2NN   |FedAVg|iid   | xxxx%        |3.6hrs    | A       | 1.0  |5  |10 | 0.01  | SGD   | 
2NN   |FedAVg|iid   | xxxx%        |1.7hrs    | A       | 0.5  |5  |10 | 0.01  | SGD   | 
2NN   |FedAVg|iid   | xxxx%        |0.7hrs    | A       | 0.2  |5  |10 | 0.01  | SGD   | 
2NN   |FedAVg|iid   | xxxx%        |xxxhrs    | A       | 0.1  |5  |10 | 0.01  | SGD   |
2NN   |FedAVg|iid   | xxxx%        |xxxhrs    | A       | 0.0  |5  |∞  | 0.01  | SGD   |
2NN   |FedAVg|iid   | xxxx%        |xxxhrs    | A       | 1.0  |5  |∞  | 0.01  | SGD   |
2NN   |FedAVg|iid   | xxxx%        |xxxhrs    | A       | 0.5  |5  |∞  | 0.01  | SGD   |
2NN   |FedAVg|iid   | xxxx%        |xxxhrs    | A       | 0.2  |5  |∞  | 0.01  | SGD   |
2NN   |FedAVg|iid   | xxxx%        |xxxhrs    | A       | 0.1  |5  |∞  | 0.01  | SGD   |

##### CNN/non-IID
Model |Method|Data  | Val test acc |Time used | Machine | Frac | E | B | Lr/O  | Optim | Status
------|------|------| --------     |--------- | --------| -----|---|---| ----- | ----- | ------
2NN   |FedAVg|N-iid | xxxx%        |xxxhrs    | A       | 0.0  |5  |10 | 0.01  | SGD   |
2NN   |FedAVg|N-iid | xxxx%        |3.6hrs    | A       | 1.0  |5  |10 | 0.01  | SGD   | 
2NN   |FedAVg|N-iid | xxxx%        |1.7hrs    | A       | 0.5  |5  |10 | 0.01  | SGD   | 
2NN   |FedAVg|N-iid | xxxx%        |0.7hrs    | A       | 0.2  |5  |10 | 0.01  | SGD   | 
2NN   |FedAVg|N-iid | xxxx%        |xxxhrs    | A       | 0.1  |5  |10 | 0.01  | SGD   |
2NN   |FedAVg|N-iid | xxxx%        |xxxhrs    | A       | 0.0  |5  |∞  | 0.01  | SGD   |
2NN   |FedAVg|N-iid | xxxx%        |xxxhrs    | A       | 1.0  |5  |∞  | 0.01  | SGD   |
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
Model |Method|Data  | Val test acc|Time used | Machine | Frac | E | B | Lr/O     | Optim | Status
------|------|------| --------    |--------  |-------- | -----|---|---| -----    | ------| -----
CNN   |FedSGD|iid   | 98.53% -200 |5.5mins   | T       | 0.1  |1  |∞  | 0.15     | SGD   | fs done
CNN   |FedAVg|iid   | 98.68% -200 |10.3mins  | T       | 0.1  |5  |∞  | 0.08     | SGD   | fs done
CNN   |FedAVg|iid   | 98.66% -200 |5.8mins   | T       | 0.1  |1  |50 | 0.2      | SGD   | fs done
CNN   |FedAVg|iid   | 98.51% -200 |28.2mins  | T       | 0.1  |20 |∞  | 0.16/0.2 | SGD   | fs done
CNN   |FedAVg|iid   | %           |hrs       | T       | 0.1  |1  |10 | 0.xx     | SGD   | 
CNN   |FedAVg|iid   | %           |hrs       | T       | 0.1  |5  |50 | 0.01     | SGD   | 
CNN   |FedAVg|iid   | 98.61% -200 |28.1mins  | T       | 0.1  |20 |50 | 0.10     | SGD   | fs done
CNN   |FedAVg|iid   | %           |hrs       | T       | 0.1  |5  |10 | 0.01     | SGD   | 
CNN   |FedAVg|iid   | %           |hrs       | T       | 0.1  |20 |10 | 0.01     | SGD   | 

##### CNN/non-IID
Model |Method|Data  | Val test acc|Time used | Machine | Frac | E | B | Lr/O  | Optim | Status
------|------|------| --------    |--------  |-------- | -----|---|---| ----- | ------| -----
CNN   |FedSGD|N-iid | %           |hrs       | T       | 0.1  |1  |∞  | 0.01  | SGD   | 
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


##### CNN/non-IID
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
