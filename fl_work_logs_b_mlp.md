# Work logs and notes on reproducing vanilla FL paper - Part B: grid search of learning rate

### Approach (fundamental principle)
* WY's MLP and CNN models is used instead of original models found in AshwinRJ's repository
* The grid search conducted below follows the approach described in Part A section III:
    1. Coarse search using a bigger resolution, a factor of 10 or 0.1, say, start from 1e-5, then 1e-4, 1e-3, 1e-2, 1e-1, and 1.0,
    2. Finer search using a smaller resolution, factor-2, i.e., {2, 4, 8}, between two best values found in the previous coarse search.
* **CAUTION** To decide the optimized learning rate, the vanilla paper took the best values achieved in all prior rounds:
> optimizing η as described above and then making each curve monotonically improving by taking the best value of test-set accuracy achieved over all prior rounds

### Approach for MNIST IID data
* Initial coarse searches in a grid {1e-4, 1e-3, 1e-2, 1e-1, 1.0}, best values were found around 1e-2 and 1e-1.
* It is found that, for WY's CNN model of learning MNIST, best learning rate are most likely to appear around {0.08, 0.1, 0.16, 0.2}. Therefore, one can reduce the search range to {0.08, 0.1, 0.2} firstly, since these three values are most likely to be the best learning rate. 0.16 can be checked additionaly.
* Most of the completed search show that WY's CNN model of learning MNIST leads to monototically increased test accuracy. Therefore, one may use 100 rounds instead of 200 rounds for searching the learning rate.

### Approach for MNIST non-IID data
* Initial coarse searches in a grid {1e-5, 1e-4, 1e-3, 1e-2, 1e-1}, best values were found around 1e-3 and 1e-2
* It is found that, for WY's CNN model of learning MNIST, best learning rate are most likely to appear around {0.008, 0.01, 0.02, 0.04}. Therefore, one can reduce the search range to {0.01, 0.02, 0.04} firstly, since these three values are most likely to be the best learning rate. 0.008, 0.06 can be checked additionaly.
* It was observed that 0.02 are most likely to be the best learning rate for non-IID cases.

### Updated approach for MNIST IID/Non-IID data
Following the vanilla FL paper, grid search resolution of $10^{1/3}$ approximately leads to multiplicative factors {1,2,5,10}
1. So for IID, since after coarse search the best values appears in 0.01-1.0, finer searches could be done within {0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0} according to the above resolution. 
    * According to the previous search trials, reasonable finer searches can be narrowed to **{0.05, 0.1, 0.2}**.
    * Since candidate lr=0.08 no longer exists and lr=0.05 is very likely not better than 0.08, so existing search results will be updated by comparison between 0.1 and 0.2.
    * (Update) lr=0.2 is likely to cause instability in real test runs. If so, finer search in {0.07, 0.1, 0.15} can be used, which is calculated as per multiplicative factors {6.8, 10, 1.5} obtained from resolution $10^(1/6)$.
2. Similarly for non-IID, best values are likely to appear around 0.001-0.1, so that finer searches could be done witin {0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1}. 
    * According to the previous search trials, reasonable finer searches can be narrowed done to **{0.01, 0.02, 0.05}**.
    * Since candidate lr=0.04 no longer exists and lr=0.06 is better than 0.04 according to existing trials, so existing search results indicating lr=0.04 can be replaced by 0.05. However, there may not be significant differences.
    * Multiplicative factors {1.5, 2.2, 3.2, 4.6, 6.8} obtained from resolution $10^(1/6)$ may also be used to generates alternative finer searches lr={0.007, 0.01, 0.015, 0.022, 0.032, 0.046}. Then, according to pervious search trials, {0.022, 0.032} can be used instead of 0.04 if instability is observed in real test run.


### Generic remarks
Some test runs are unstable under searched learning rate, this is
* either because the learning rate is too close to instability region
* or due to the limited memory and computational power of the machine 

### Experiment 1: increase parallism
* Lr/O means the optimized value of learning rate
* The learning rate found using coarse search will be marked as "done cs"
* The learning rate found using further finer search will be marked as "done fs"
* Grid searches are conducted using a validation dataset which is 20% of the original MNIST training set, then train the model over 200 or 400 rounds and compare the test acc to determine the best values.

#### CNN/IID
* Default number of rounds is 200, if any run differs, then it will be marked as XX.XX%-XXX
* Compare the highest test acc achieved in all rounds.
    1. If two lr ties or differs little in test acc, take the one with smoother test loss and test acc curves.
    2. If the lr achieve higher maximum test acc, but the test loss and test acc curves exhibits high instablity, then take the one with 2nd highes maximum test acc. 

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

##### Remarks
* asdf

#### CNN/non-IID
* Default number of rounds is 400, if any run differs, then it will be marked as XX.XX%-XXX
* Compare the highest test acc achieved in all rounds.
    1. If two lr ties or differs little in test acc, take the one with smoother test loss and test acc curves.
    2. If the lr achieve higher maximum test acc, but the test loss and test acc curves exhibits high instablity, then take the one with 2nd highes maximum test acc. 

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

##### Remarks
1. asdf



### Experiment 2: increase local computation
* The fraction number is fixed at C=0.1
* The learning rate found using coarse search will be marked as "done cs"
* The learning rate found using further finer search will be marked as "done fs"
* Grid searches are conducted using a validation dataset which is 20% of the original MNIST training set, then train the model over 200 or 400 rounds and compare the test acc to determine the best values.

#### CNN/IID
* Default number of rounds is 200, if any run differs, then it will be marked as XX.XX%-XXX
* Lr/O is ranked as per alternative search training 20 rounds on the full dataset and test on test set, original best search results is marked by *, status marked as re-rank

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

##### Remarks
1. asdf

#### CNN/non-IID
* Default number of rounds is 400, if any run differs, then it will be marked as XX.XX%-XXX
* Compare the highest test acc achieved in all rounds.
    1. If two lr ties or differs little in test acc, take the one with smoother test loss and test acc curves.
    2. If the lr achieve higher maximum test acc, but the test loss and test acc curves exhibits high instablity, then take the one with 2nd highes maximum test acc. 
    
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

##### Remarks
1. asdf

