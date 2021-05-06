# Work logs and notes on reproducing vanilla FL paper - Part C-2: grid search of learning rate

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

### Approach for CIFAR fixed E,B (tf cnn)
* Grid searches of initial learning rate is conducted prior to the learning rate decay. 
* The searches of best initial lr are conducted using the entire original CIFAR10 training set, then train the model over a relatively shorter rounds, say 100-500 rounds, and compare the test acc to determine the best values.
* Learning rate of SGD, FedSGD, FedAvg are first seachred in {1e-5, ..., 1.0}, 
    * For SGD, it was found that best values are around {0.01, 0.1}, 
    * Then finer search in {0.01, 0.15, 0.02, 0.03, 0.05, 0.07, 0.1 ,0.15, 0.22} obtained from resolution factors {1, 1.5, 2.2, 3.2, 4.6, 6.8, 10} show that the best initial lr values are likely be within {0.01, 0.015, 0.02, 0.03}
* Adopting the similar grid-search multiplicative factors, the learning-rate decay can be searched in {0.9910, .9915, .9922, .9932, .9946, .9968,} with smaller resolution or {.991, 992, 995} with larger resolution.


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
* Grid searches are conducted using a validation dataset which is 20% of the original MNIST training set, then train the model over 200 or 400 rounds and compare the test acc to determine the best values.

##### CNN/IID
* Default number of rounds is 200, if any run differs, then it will be marked as XX.XX%-XXX
* If the test acc at final round differs little, then check the final 10 rounds and choose the one with higher average value.
* If two lr ties in test acc, 
    1. 0.08 ties with 0.1, then take 0.1, to use larger ones for quicker convergence hopefully 
    2. 0.1 ties with 0.2, then take 0.1, to use smaller ones for better stability hopeflly 
    3. 0.08 ties with 0.2, then take 0.08, to use smaller ones for better stability hopeflly 

Model |Method|Data  | Val test acc |Time used | Machine | Frac | E | B | Lr/O      | Optim | Status
------|------|------| --------     |--------- | --------| -----|---|---| -----     | ----- | ------
CNN   |FedAVg|iid   | 98.41%       |0.09hrs   | A       | 0.0  |5  |10 | 0.1/0.08  | SGD   | fs done
CNN   |FedAVg|iid   |              |3.6hrs    | A       | 1.0  |5  |10 |           | SGD   | cancelled
CNN   |FedAVg|iid   | 98.49%       |0.56hrs   | A       | 0.5  |5  |10 | 0.1       | SGD   | fs done
CNN   |FedAVg|iid   | 98.50%       |0.27hrs   | A       | 0.2  |5  |10 | 0.1       | SGD   | fs done
CNN   |FedAVg|iid   | 98.43%       |0.17hrs   | A       | 0.1  |5  |10 | 0.08/0.1  | SGD   | fs done
CNN   |FedAVg|iid   | 98.36%       |0.09hrs   | A       | 0.0  |5  |∞  | 0.1       | SGD   | fs done
CNN   |FedAVg|iid   |              |xxxhrs    | A       | 1.0  |5  |∞  |           | SGD   | cancelled
CNN   |FedAVg|iid   | 98.46%       |0.95hrs   | A       | 0.5  |5  |∞  | 0.1/0.2   | SGD   | fs done
CNN   |FedAVg|iid   | 98.61%       |0.27hrs   | A       | 0.2  |5  |∞  | 0.2       | SGD   | fs done
CNN   |FedAVg|iid   | 98.45%       |0.18hrs   | A       | 0.1  |5  |∞  | 0.2       | SGD   | fs done

##### Remarks
* For {E=5,B=10,C=0.1}, lr=0.2 is unstable in real test after about 190 rounds, so lr=0.2 is discarded.
* For {E=5,B=10,C=0.2}, lr=0.2 leads to best performance in validation run. But it is sometimes unstable in real test even at the beginning round. lr=0.15 is validated, the results differs little with that of lr=0.1. Considering the stability boundary, so lr=0.1 is recommended over lr=0.2, 0.15.
* For {E=1, B=10}, lr=0.2 and 0.15 is unstable in real test run, try {0.1, 0.07} instead. The other parameter combinations can follow similar approach for dealing instability under large lr.

##### CNN/non-IID
* Default number of rounds is 400, if any run differs, then it will be marked as XX.XX%-XXX
* Compare the highest test acc achieved in all rounds.
    1. If two lr ties or differs little in test acc, take the one with smoother test loss and test acc curves.
    2. If the lr achieve higher maximum test acc, but the test loss and test acc curves exhibits high instablity, then take the one with 2nd highes maximum test acc. 


*Watch out the "FAKE NEWS": in {0.02, 0.04, 0.06}, larger lr may give better final stage test acc, but is also more likely to present instability. If the difference between test acc is not significant, then it could be safer to use smaller ones, e.g., 0.04 over 0.06*.

Model |Method|Data  | Val test acc |Time used | Machine | Frac | E | B | Lr/O  | Optim | Status
------|------|------| --------     |--------- | --------| -----|---|---| ----- | ----- | ------
CNN   |FedAVg|N-iid | 96.61%       |0.18hrs   | A       | 0.0  |5  |10 | 0.04  | SGD   | fs done
CNN   |FedAVg|N-iid |              |          | A       | 1.0  |5  |10 |       | SGD   | cancelled
CNN   |FedAVg|N-iid | 96.93%       |xxxhrs    | T       | 0.5  |5  |10 | 0.04  | SGD   | fs done
CNN   |FedAVg|N-iid | 96.85%       |0.55hrs   | A       | 0.2  |5  |10 | 0.04  | SGD   | fs done
CNN   |FedAVg|N-iid | 97.13%       |0.35hrs   | A       | 0.1  |5  |10 | 0.04  | SGD   | fs done
CNN   |FedAVg|N-iid | 95.83%       |xxxhrs    | T       | 0.0  |5  |∞  | 0.02  | SGD   | fs done
CNN   |FedAVg|N-iid |              |          | A       | 1.0  |5  |∞  |       | SGD   | cancelled
CNN   |FedAVg|N-iid | 96.98%       |0.99hrs   | T       | 0.5  |5  |∞  | 0.04  | SGD   | fs done
CNN   |FedAVg|N-iid | 96.47%       |0.56hrs   | T       | 0.2  |5  |∞  | 0.04  | SGD   | fs done
CNN   |FedAVg|N-iid | 96.76%       |0.35hrs   | A       | 0.1  |5  |∞  | 0.04  | SGD   | fs done
##### Remarks
1. In the above table, 0.04 appears to be the best learning rates for most of the parameter combinations


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
* Grid searches are conducted using a validation dataset which is 20% of the original MNIST training set, then train the model over 200 or 400 rounds and compare the test acc to determine the best values.

##### CNN/IID
* Default number of rounds is 200, if any run differs, then it will be marked as XX.XX%-XXX
* Lr/O is ranked as per alternative search training 20 rounds on the full dataset and test on test set, original best search results is marked by *, status marked as re-rank

Model |Method|Data  | Val test acc|Time used | Machine | Frac | E | B | Lr/O               | Optim | Status
------|------|------| --------    |--------  |-------- | -----|---|---| -----              | ------| -----
CNN   |FedSGD|iid   | 98.54%      |5.5mins   | T       | 0.1  |1  |∞  | 0.2/0.15           | SGD   | fs done
CNN   |FedAVg|iid   | 98.68%      |10.3mins  | T       | 0.1  |5  |∞  | 0.2/0.15/0.1*      | SGD   | re-rank
CNN   |FedAVg|iid   | 98.66%      |5.8mins   | T       | 0.1  |1  |50 | 0.2                | SGD   | fs done
CNN   |FedAVg|iid   | 98.51%      |28.2mins  | T       | 0.1  |20 |∞  | 0.15*/0.1/0.2      | SGD   | re-rank
CNN   |FedAVg|iid   | 98.51%      |5.8mins   | T       | 0.1  |1  |10 | 0.2/0.15           | SGD   | fs done
CNN   |FedAVg|iid   | 98.48%      |10.5mins  | T       | 0.1  |5  |50 | 0.2/0.15           | SGD   | fs done
CNN   |FedAVg|iid   | 98.61%      |28.1mins  | T       | 0.1  |20 |50 | 0.15/0.2/0.1*      | SGD   | re-rank
CNN   |FedAVg|iid   | 98.50%      |10.5mins  | T       | 0.1  |5  |10 | 0.15/0.2*/0.1      | SGD   | re-rank
CNN   |FedAVg|iid   | 98.55%      |46.7mins  | T       | 0.1  |20 |10 | 0.1/0.15/0.07/0.2* | SGD   | fs done

##### Remarks
1. For {E=1, B=10}, lr=0.2 and 0.15 is unstable in real test run, try {0.1, 0.07} instead.
2. For {E=5, B=50}, lr=0.2 is unstable in real test run after 297 rounds, try {0.15, 0.1, 0.07} instead.

##### CNN/non-IID
* Default number of rounds is 400, if any run differs, then it will be marked as XX.XX%-XXX
* Compare the highest test acc achieved in all rounds.
    1. If two lr ties or differs little in test acc, take the one with smoother test loss and test acc curves.
    2. If the lr achieve higher maximum test acc, but the test loss and test acc curves exhibits high instablity, then take the one with 2nd highes maximum test acc. 
    
Model |Method|Data  | Val test acc |Time used | Machine | Frac | E | B | Lr/O       | Optim | Status
------|------|------| --------     |--------  |-------- | -----|---|---| -----      | ------| -----
CNN   |FedSGD|N-iid | 97.17%       |0.20hrs   | T       | 0.1  |1  |∞  | 0.04       | SGD   | fs done
CNN   |FedAVg|N-iid | 96.04%/96.04%|0.35hrs   | A       | 0.1  |5  |∞  | 0.02       | SGD   | fs done
CNN   |FedAVg|N-iid | 97.15%       |0.19hrs   | T       | 0.1  |1  |50 | 0.04       | SGD   | fs done
CNN   |FedAVg|N-iid | 96.96%       |0.93hrs   | T       | 0.1  |20 |∞  | 0.04       | SGD   | fs done
CNN   |FedAVg|N-iid | 96.67%       |0.20hrs   | T       | 0.1  |1  |10 | 0.04       | SGD   | fs done
CNN   |FedAVg|N-iid | 97.04%       |0.35hrs   | A       | 0.1  |5  |50 | 0.04       | SGD   | fs done
CNN   |FedAVg|N-iid | 97.34%       |0.95hrs   | T       | 0.1  |20 |50 | 0.04       | SGD   | fs done
CNN   |FedAVg|N-iid | 96.52%       |0.35hrs   | A       | 0.1  |5  |10 | 0.04       | SGD   | fs done
CNN   |FedAVg|N-iid | 97.06%       |0.91hrs   | T       | 0.1  |20 |10 | 0.04       | SGD   | fs done
##### Remarks
1. In the above table, 0.04 appears to be the best learning rates for most of the parameter combinations


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


#### C. Experiment 3: CIFAR10 learning performance

##### 3-A: Speed up of convergence agaisnt comm. round, FedAvg vs FedSGD vs SGD
* Fraction of users is fixed at C=0.1
* FedSGD and FedAVg use fixed E=5, and FedAvg use fixed B=50
* For SGD, batch size is fixed at B=100, so number of mini-batch updates is also $500R$ since N=50,000 so mini-batch update per round is N/B=500.
* Since E, B are fixed then total number of mini-batch updates per participant is $n=R\times \frac{NE}{100B}=50R$, where N=50,000 is the total amount of data.
* FedSGD do only one mini-batch update per local epoch which means the total mini-batch updates is $5R$ per participant.
* Therefore, every 100,000 rounds mini-batch updates means 
    1. 200 epochs for SGD, 
    2. 200 comm. rounds for FedAvg, since each round there are 10 clients each making 50 mini-batch update 
    3. 20,00 comm. rounds for FedSGD, since each round there are 10 clients each making 5 mini-batch udpate
* One cannot afford to perform that many rounds for FedSGD, a reasonable approach is to let FedAvg and FedSGD perform identical rounds of learning, as what are performed in the previous two experiments, e.g., 100-200 rounds.
    * Considering either *torch cnn* and *tf cnn* overfits in less than 100 epochs of SGD, one may perform 100 rounds FedAvg/FedSGD
    * Reduced rounds of experiment is a make-do method, it is acceptable for the time being as long as the experiment result can reflect the same fundamental conclusion drawn in the vaniila FL paper
* Decay means the learning-rate decay
* **TO-DO on 5-th May**:
    - [ ] Test SGD with lr={2e-5, 5e-5, 1e-4} for 10000 rounds (running on A ... ).


Model |Method|Data  | T Rnd |Time      | Machine | Frac | E | B | Lr/O      |Decay  | Optim | Status
------|------|------| ----  |--------  | -----   |---   |---| - | -----     |------ | ----- | ------
CNN   |SGD   |iid   | xxxx  |hrs       | T       |      |   |100| 0.01@dp   | SGD   |       | run on A
CNN   |FedSGD|iid   | 8000  |~8hrs     | A       | 0.1  |1  |∞  | 0.1@dp    |       | SGD   | done
CNN   |FedAVg|iid   | 4000  |9.18hrs   | T       | 0.1  |5  |50 | 0.03@dp   |       | SGD   | done

##### Remarks
1. Using "dropout" for *tf cnn* model, 
    * FedAvg with lr=0.03, 0.05 effectively reduced overfitting within 200 rounds, reaching a maximum test acc around 72%
    * SGD with lr=0.07 and 0.1 also exhibits effectiveness of handling overfitting within 100 rounds. However, SGD with lr=0.03, 0.05 still suffer from severe overfitting in 200 rounds, there is no improvement over the curves obtained without dropout layer. It is not clear whether some error occurred in the above SGD runs with lr=0.03 and 0.05. 
    * **Experiments with 200 rounds/epochs show that**:
        1. For FedAvg, best lr could be 0.1 among lr={0.03, 0.05, 0.07, 0.1}, larger values do not improve further.
        2. For SGD, lr=0.01 looks best. However, for each tested lr in {0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2}, the resulted learning curve is similar to that generated without dropout, i.e, ***overfitting still exists. It is suspected that the learning rate should be tuned even smaller, like 0.001***.
    * However, **experiments with 2000 rounds show that**:
        1. For SGD, for lr in {2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 1.5e-2, 2e-2, 3e-2}, larger lr converge to 72% test acc quicker but tends to overfit earlier and needs the "ealier stopping". ***Among all lr, lr=0.01 reaches 72% around 45 rounds and achieves a maximum test acc about 74%***. Futher tests using learning-rate decay of 0.99 does not show any improvement for SGD with lr=0.01 in 200 rounds. The above observations suggest that even smaller lr like {5e-5, 1e-4, ..., 1e-3} may reach a higer test acc in long-term runs over, say, 10000 rounds.
        2. For FedAvg, larger lr in {0.05,0.07,0.1} converge quicker before 200 rounds, ***however, lr=0.05 outperforms and converges to 0.72 quicker afterwards***.
        3. For FedSGD, coarse searches in {0.05, 0.1, 0.2} find that lr=0.2 keeps increasing in test acc in 4000 rounds (3.8 hrs), and is better than smaller lr=0.1,0.05. So, next searches could be within {0.1, 0.2, 0.5} over 8000 rounds.
    * Tests over 4000/8000 rounds show that:
        1. For FedAvg, tests of lr={0.005, 0.01, 0.03, 0.05} in 4000 rounds show that ***lr=0.03 reach a higher test acc***. Both lr=0.05 and lr=0.01 converge in slower speed and to lower test acc, lr=0.05 converges a little bit quicker than lr=0.03 but it converges to a similar test acc as lr=0.01. None of {0.005,0.01,0.05} reach 74%. 
        2. For FedSGD, test of lr={0.1, 0.2, 0.5} for 8000 rounds show that the larger lr the quicker the convergence, but ***lr=0.1 reach the highest test acc among these three lr***, and the related test loss is just about to rise again (just not to overfit), which suggests that in 8000 rounds, even smaller lr like 0.05, 0.03, and 0.02 may not achieve same test acc and also converge slower compared with lr=0.1. 
2. If the target is to compare speedup for a specific test acc target, then one may not need to care too much about how to reach a higest final test acc for each algorithm. 
    * Instead, one may wish to tune the quickest fashion for the convergence of each algorithm, and then do the comparison of speedup.
    * Limited computational power and project timeline does not allow for
        1. Finer searches of best learning rate (and decay scheme) for each algorithm.
        2. Long-term runs (e.g., 200,000 rounds) of SGD in order to obtain the highest possible test acc is not fesible for the time being.
    * The convergence objective can be set to 0.68, 0.7, 0.72, 0.74


##### 3-B: Per mini-batch update convergence FedAvg vs SGD 
* 300,000 rounds mini-batch updates used in the vanilla FL paper is too many to complete in the allowed time for now. Therefore, one may consider **100,000 mini-batch updates** instead, which is equivalent to
    1. 200 rounds (B=100) or for **100 rounds (B=50) SGD**,
    2. 200 rounds for FedAvg E=5, B=50, C=0.1
    3. 100 rounds for FedAvg E=10, B=50, C=0.1, 50 rounds E=20, B=50, C=0.1, 20 rounds for E=5, B=50, C=1.0
    4. 1000 rounds for FedAvg E=1, B=50, C=0.1
    5. 2000 rounds for FedAvg E=5, B=50, C=0.0
* The learning rates of every setting optimized to produce highest possible test acc over 100,000 mini-batch updates, while sacrificing the convergence speed as least as possible.


Model |Method|Data  | T Rnd |Time      | Machine | Frac | E | B | Lr/O         | Optim | Status
------|------|------| ----  |--------  | -----   |---   |---| - | -----        | ----- | ------
CNN   |SGD   |iid   | 100   |hrs       | T       |      |   |50 | 0.0032@dp    | SGD   | 
CNN   |FedAVg|iid   | 2000  |1.2-1.6hrs| T       | 0.0  |5  |50 | 0.02/0.32@dp | SGD   | done
CNN   |FedAVg|iid   | 1000  |hrs       | T       | 0.1  |1  |50 | 0.1@dp       | SGD   | done
CNN   |FedAVg|iid   | 200   |hrs       | T       | 0.1  |5  |50 | 0.1@dp       | SGD   | done
CNN   |FedAVg|iid   | 100   |hrs       | T       | 0.1  |10 |50 | 0.2@dp       | SGD   | done
CNN   |FedAVg|iid   | 50    |hrs       | T       | 0.1  |20 |50 | 0.2@dp       | SGD   | done
CNN   |FedAVg|iid   | 20    |hrs       | T       | 1.0  |5  |50 | 0.32@dp      | SGD   | done

##### Remarks
1. It has been tested that for SGD, learning curve with B=50 varies little compared with that obtained with B=100, so learning rate for SGD in this test could be same as the one in the previous experiment.
2. For SGD with B=50, among lr={0.002, 0.0032, 0.0046, 0.005, 0.01, 0.02}, lr=0.1 converge to 73% test acc quickest (within 40 rounds), but smaller lrs reach a slightly higher maximum test acc. One can choose to show 
    * lr=0.01 with early stopping @40 rounds; or
    * lr=0.005 with early stopping @80 rounds; or
    * lr=0.0032 with 100-round full run
3. For FedAvg E=5, B=50, C=0.0, among lr={0.01, 0.02, 0.032, 0.05, 0.1, 0.2}, lr=0.02 and lr=0.32 outperform the others, one can choose to show between these two runs.
4. For FedAvg E=1, B=50, C=0.1, lr=0.1 works better than 0.05 and 0.2 within 1000 rounds.
5. For FedAvg E=5, B=50, C=0.1, lr=0.1 works better than 0.05 within 200 rounds (number of rounds that this parameter combination needs to do for 100,000 mini-batch updates for this experiment).
6. For FedAvg E=10, B=50, C=0.1, tests over 100 rounds show that lr=0.2 converges achieve highest test acc and converges quickest among lr={0.001, 0.01, 0.1, 0.2, 0.32, 0.5, 0.7}.
7. For FedAvg E=20, B=50, C=0.1, tests over 50 rounds show that lr=0.2 converges achieve highest test acc and converges quickest among lr={0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 0.7}.
8. For FedAvg E=5, B=50, C=1.0, larger lr in {0.001, 0.01, 0.1, 0.2, 0.32} achieve higher test acc in 20 round and also converges quicker, lr=0.5 converges even slower than lr=0.1 and become unstable after 10 rounds.



#### C. Experiment 4: Additional FedAvg with very large E and unbalanced non-IID data
Consider these additional experiment over MNIST if time permits (on 5/6-th May):
- [ ] test FedAVg E={100,200,500} B=10 C=0.1 and compare with E=1 over IID/non-IID data, using lr=0.1 (the same as E=1 B=10 C=0.1) to see the effect of very large amount of local computation.
- [ ] test FedAVg E=1 B=10 C=0.1 over unbalanced non-IID data, and compare with IID, balanced-non-IID, may use the same lr as FedAvg E=1 B=10 C=0.1 in  balanced-IID 




#### D. Training time summary
Model |Method| Data |Time/rnd | 100-rnd time    | Machine |Frac | E | B | Lr    | Optim
------|------|------|-------- | --------------  |-------- |-----|---|---| ----- | ---------
CNN   |FedSGD|M-iid | 2.2s    | 3.7mins/0.06hrs | T       |0.1  |1  | ∞ | 0.01  | SGD
CNN   |FedAVg|M-iid | 5.6s    | 9.3mins/0.16hrs | T       |0.1  |5  | ∞ | 0.01  | SGD
CNN   |FedAvg|M-iid | 129.12s | 3.6hrs          | A       |1.0  |5  |10 | 0.01  | SGD 
CNN   |FedAvg|M-iid | 39.4s   | 1.1hrs          | A       |0.1  |20 |10 | 0.01  | SGD
CNN   |FedAvg|M-iid | 25.2s   | 42mins/0.7hrs   | A       |0.2  |5  |10 | 0.01  | SGD
CNN   |FedAvg|M-iid | 42.34s  | 1.2hrs          | T       |0.1  |20 |10 | 0.01  | SGD
CNN   |FedAvg|M-iid | 2.2s    | 3.6mins/0.06hrs | T       |0.1  |1  |50 | 0.01  | SGD
