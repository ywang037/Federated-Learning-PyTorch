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

### Approach for CIFAR fixed E,B (tf cnn)
* Grid searches of initial learning rate is conducted prior to the learning rate decay. 
* The searches of best initial lr are conducted using the entire original CIFAR10 training set, then train the model over a relatively shorter rounds, say 100-500 rounds, and compare the test acc to determine the best values.
* Learning rate of SGD, FedSGD, FedAvg are first seachred in {1e-5, ..., 1.0}, 
    * For SGD, it was found that best values are around {0.01, 0.1}, 
    * Then finer search in {0.01, 0.15, 0.02, 0.03, 0.05, 0.07, 0.1 ,0.15, 0.22} obtained from resolution factors {1, 1.5, 2.2, 3.2, 4.6, 6.8, 10} show that the best initial lr values are likely be within {0.01, 0.015, 0.02, 0.03}
* Adopting the similar grid-search multiplicative factors, the learning-rate decay can be searched in {0.9910, .9915, .9922, .9932, .9946, .9968,} with smaller resolution or {.991, 992, 995} with larger resolution.

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

#### CNN/non-IID
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



### Experiment 2: increase local computation
* The fraction number is fixed at C=0.1
* The learning rate found using coarse search will be marked as "done cs"
* The learning rate found using further finer search will be marked as "done fs"
* Grid searches are conducted using a validation dataset which is 20% of the original MNIST training set, then train the model over 200 or 400 rounds and compare the test acc to determine the best values.

#### CNN/IID
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

#### CNN/non-IID
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



### Experiment 3: CIFAR10 learning performance

#### Baseline SGD using data augmentation and model with batch normalization
1. The baseline SGD benchmark is same for FedAvg/FedSGD in both IID and non-IID cases.
2. It has been founded that: 
    - adding batch normalization alone (after relu) to model *tf cnn* can improve the test accuarcy remarkably.
    - using data augmentation is very effective in regularization, which can drastically improve the generalization ability as well as test accuracy
3. Up to 400 epochs, it has been observed that:
    - the test run with batch normalization alone and `t1` transform has clear advantages in both convergance rate and highest test accuracy, which suggests that the most favorable model could be the one with batch normalization alone, i.e., `wycnn_tfbn` and most favorable data augmentation could be transform `t1`.
    - The model uses both batch normalization and dropout performs inferior to the one uses batch normalization only, but is still obviously better than the one with dropout only.
    - The original `tf_cnn` with `t1` transform perform even better than `wycnn_dp` and `wycnn_bndp` in test accuracy, only slighly inferior to `wycnn_bndp` in convergence rate before the first 100 epochs. This indicates that with data augmentation transform `t1`, it is no longer necessary to use the dropout layer, which even cause performance degradation in this case.
    - The learning curve resulted from `wycnn_bn` with `t1` is less noisy than obtaiend by `tf_cnn` and `t1`. 
    - Using `t2` or `t1` has no distinct difference: `t2` works slightly better for `wycnn_bn` whereas `t1` works slightly better for `tf_cnn`. 
    - `t2` takes 60% and 92% more time to train than using `t1` in 200 and 400 epochs, respectively. 
    - All runs with data augmentation `t1` or `t2`(no matter which model is used) do not tend to overfit up to 400 epochs.

Model       |Method|Data Augmentation                 | Test acc (max) | Epoch |Time      | Machine | B  | Lr/O      | Optim | Decay |  Status
-------     |------|------------------------------    |--------------- | ----  |--------  | -----   | -- | -----     |------ | ----- | ------
`wycnn_dp`  |SGD   |t0: default                       | 73.8%          | 200   |hrs       | T       |100 | 0.01@dp   | SGD   |       | benchmark
`wycnn_dp`  |SGD   |t1: mean, std, crop, flip         | 80.0%          | 200   |1.05hrs   | T       |100 | 0.01@dp   | SGD   |       | done
`wycnn_dp`  |SGD   |t2: mean, std, crop, flip, color  | 79.4%          | 200   |1.61hrs   | T       |100 | 0.01@dp   | SGD   |       | done
`wycnn_dp`  |SGD   |t1: mean, std, crop, flip         | 82.3%          | 400   |2.09hrs   | T       |100 | 0.01@dp   | SGD   |       | done
`wycnn_bn`  |SGD   |t1: mean, std, crop, flip         | 84.5%          | 200   |1.10hrs   | T       |100 | 0.01@dp   | SGD   |       | done
`wycnn_bn`  |SGD   |t1: mean, std, crop, flip         | 85.0%          | 400   |2.15hrs   | T       |100 | 0.01@dp   | SGD   |       | done
`wycnn_bn`  |SGD   |t2: mean, std, crop, flip, color  | **85.2%**      | 400   |3.55hrs   | A       |100 | 0.01@dp   | SGD   |       | done
`wycnn_bndp`|SGD   |t1: mean, std, crop, flip         | 83.3%          | 400   |2.17hrs   | T       |100 | 0.01@dp   | SGD   |       | done
`tf_cnn`    |SGD   |t1: mean, std, crop, flip         | 84.2%          | 400   |1.85hrs   | T       |100 | 0.01@dp   | SGD   |       | done, `selected`
`tf_cnn`    |SGD   |t2: mean, std, crop, flip, color  | 84.2%          | 400   |3.54hrs   | A       |100 | 0.01@dp   | SGD   |       | done

#### 3-A: Speed up of convergence agaisnt comm. round, FedAvg vs FedSGD vs SGD
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

#### 3-A1 CNN/IID
Model |Method|Data  | T Rnd |Time      | Machine | Frac | E | B | Lr/O      |Decay  | Optim | Status
------|------|------| ----  |--------  | -----   |---   |---| - | -----     |------ | ----- | ------
CNN   |SGD   |iid   | 200   |hrs       | T       |      |   |100| 0.01@dp   | SGD   |       | run on A
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
        1. For SGD, lr={2e-5, 5e-5, 7e-5, 1e-4} for 4000-5000 rounds **has not achieved test accuracy higher than 0.72**.
        2. For FedAvg, tests of lr={0.005, 0.01, 0.03, 0.05} in 4000 rounds show that ***lr=0.03 reach a higher test acc***. Both lr=0.05 and lr=0.01 converge in slower speed and to lower test acc, lr=0.05 converges a little bit quicker than lr=0.03 but it converges to a similar test acc as lr=0.01. None of {0.005,0.01,0.05} reach 74%. 
        3. For FedSGD, test of lr={0.1, 0.2, 0.5} for 8000 rounds show that the larger lr the quicker the convergence, but ***lr=0.1 reach the highest test acc among these three lr***, and the related test loss is just about to rise again (just not to overfit), which suggests that in 8000 rounds, even smaller lr like 0.05, 0.03, and 0.02 may not achieve same test acc and also converge slower compared with lr=0.1. 
2. If the target is to compare speedup for a specific test acc target, then one may not need to care too much about how to reach a higest final test acc for each algorithm. 
    * Instead, one may wish to tune the quickest fashion for the convergence of each algorithm, and then do the comparison of speedup.
    * Limited computational power and project timeline does not allow for
        1. Finer searches of best learning rate (and decay scheme) for each algorithm.
        2. Long-term runs (e.g., 200,000 rounds) of SGD in order to obtain the highest possible test acc is not fesible for the time being.
    * The convergence objective can be set to 0.68, 0.7, 0.72, 0.74


#### 3-A2 CNN/non-IID
Model      |Method|Data    | T Rnd |Time      | Machine | Frac | E | B | Lr/O      |Decay  | Optim | Augmentation   |Status
---------- |------|------- | ----  |--------  | -----   |---   |---| - | -----     |------ | ----- | -----------    |------
`tf_cnn`   |SGD   |iid     | 400   |hrs       | A       |      |   |100| 0.2       |       | SGD   | `t1` transform | done
`tf_cnn`   |SGD   |iid     | 400   |hrs       | A       |      |   |100| 0.1       |       | SGD   | `t1` transform | done
`tf_cnn`   |SGD   |iid     | 400   |hrs       | A       |      |   |100| 0.05      |       | SGD   | `t1` transform | done, `selected`
`tf_cnn`   |SGD   |iid     | 400   |hrs       | A       |      |   |100| 0.02      |       | SGD   | `t1` transform | done
`tf_cnn`   |SGD   |iid     | 400   |hrs       | A       |      |   |100| 0.01      |       | SGD   | `t1` transform | done
`tf_cnn`   |SGD   |iid     | 400   |hrs       | A       |      |   |100| 0.005     |       | SGD   | `t1` transform | done
`wycnn_bn` |FedSGD|non-iid | 8000  | <12hrs   | A       | 0.1  |1  |∞  | 0.1       |       | SGD   | `t1` transform | done
`wycnn_bn` |FedSGD|non-iid | 8000  | 8.2hrs   | A       | 0.1  |1  |∞  | 0.05      |       | SGD   | `t1` transform | done
`tf_cnn`   |FedSGD|non-iid | 8000  | 8.4hrs   | A       | 0.1  |1  |∞  | 0.1       |       | SGD   | `t1` transform | done
`tf_cnn`   |FedSGD|non-iid |30000  |33.0hrs   | A       | 0.1  |1  |∞  | 0.1       |       | SGD   | `t1` transform | done
`tf_cnn`   |FedSGD|non-iid |20000  |19.9hrs   | A       | 0.1  |1  |∞  | 0.05      |       | SGD   | `t1` transform | done
`tf_cnn`   |FedSGD|non-iid |30000  |32.5hrs   | A       | 0.1  |1  |∞  | 0.05      |       | SGD   | `t1` transform | done
`tf_cnn`   |FedSGD|non-iid |40000  |    hrs   | A       | 0.1  |1  |∞  | 0.05      |       | SGD   | `t1` transform | 
`tf_cnn`   |FedSGD|non-iid | 8000  | 8.3hrs   | A       | 0.1  |1  |∞  | 0.05      |       | SGD   | `t1` transform | done, `selected`
`tf_cnn`   |FedSGD|non-iid | 8000  |10.6hrs   | A       | 0.1  |1  |∞  | 0.02      |       | SGD   | `t1` transform | done
`tf_cnn`   |FedSGD|non-iid |40000  |    hrs   | A       | 0.1  |1  |∞  | 0.02      |       | SGD   | `t1` transform | 
`wycnn_bn` |FedAvg|non-iid | 4000  | ~14hrs   | A       | 0.1  |5  |50 | 0.1       |       | SGD   | `t1` transform | done
`tf_cnn`   |FedAvg|non-iid | 4000  | ~14hrs   | A       | 0.1  |5  |50 | 0.1       |       | SGD   | `t1` transform | done
`tf_cnn`   |FedAvg|non-iid | 4000  |13.6hrs   | A       | 0.1  |5  |50 | 0.05      |       | SGD   | `t1` transform | done, `selected`
`tf_cnn`   |FedAvg|non-iid |10000  | ~40hrs   | A       | 0.1  |5  |50 | 0.05      |       | SGD   | `t1` transform | done
`tf_cnn`   |FedAvg|non-iid | 4000  |14.1hrs   | A       | 0.1  |5  |50 | 0.02      |       | SGD   | `t1` transform | done
`tf_cnn`   |FedAvg|non-iid |10000  |    hrs   | A       | 0.1  |5  |50 | 0.02      |       | SGD   | `t1` transform | 
`tf_cnn`   |FedAvg|non-iid | 4000  |13.6hrs   | A       | 0.1  |5  |50 | 0.01      |       | SGD   | `t1` transform | done
`tf_cnn`   |FedAvg|non-iid |10000  | ~40hrs   | A       | 0.1  |5  |50 | 0.01      |       | SGD   | `t1` transform | terminated at 9K rounds
`tf_cnn`   |FedAvg|non-iid |20000  |    hrs   | A       | 0.1  |5  |50 | 0.005     |       | SGD   | `t1` transform | 
`tf_cnn`   |FedAvg|non-iid | 4000  |20.0hrs   | A       | 0.1  |5  |50 | 0.05      |       | SGD   | `t2` transform | done
`tf_cnn`   |FedAvg|non-iid |20000  |    hrs   | A       | 0.1  |1  |10 | 0.1       |       | SGD   | `t1` transform | unstable
`tf_cnn`   |FedAvg|non-iid |20000  |27.2hrs   | A       | 0.1  |1  |10 | 0.05      |       | SGD   | `t1` transform | rone
`tf_cnn`   |FedAvg|non-iid |20000  |27.2hrs   | A       | 0.1  |1  |10 | 0.02      |       | SGD   | `t1` transform | done
`tf_cnn`   |FedAvg|non-iid |20000  |    hrs   | A       | 0.1  |5  |10 | 0.05      |       | SGD   | `t1` transform | unstable
`tf_cnn`   |FedAvg|non-iid |20000  |94.0hrs   | A       | 0.1  |5  |10 | 0.02      |       | SGD   | `t1` transform | done
`tf_cnn`   |FedAvg|non-iid |20000  |    hrs   | A       | 0.1  |5  |10 | 0.01      |       | SGD   | `t1` transform | run on A 

##### Remarks
1. Further baseline test runs show that the alternative transform leads to better performance in both convergence rate and test acc when using the same lr as standard transform. However, in this series of tests, the standard transform is still used for the consistency.
2. For these non-IID tests regarding CIFAR10, the baseline SGD is the same as IID tests.
3. All the test runs referred below and shown in the above table use data augmentation `t1`. 
4. For SGD with runs up to 400 epochs:
    - `lr=0.1` has quickest convergence in first 50 epochs, `lr=0.05` has the quickest convergence in first 250 epochs, `lr=0.01` has highest test accuracy in 400 epochs.
5. For FedAvg 
    * with runs up to 4000 rounds:
        - It has been observed that in these non-IID tests, the learning curves oscillates a lot more than those in IID cases, for all the lr tried.
        - Larger lr in {0.005, 0.01, 0.02, 0.05, 0.1} performs better up to 2K rounds.
        - Test run with `wycnn_bn` outperform `wycnn_dp` in the first 2K rounds but then become similarly bad. This run also seems to severely overfit after 1K rounds.
        - **Unlike the case of SGD** (where data is not non-IID), in this non-IID setting, performance of FedAvg using `wycnn_bn` is much worsen than using `tf_cnn`
        - Using `tf_cnn` with `lr=0.05` and data augmentation produces best results where `t2` performs similarly as `t1` up to 4K rounds, however `t2` takes about 47% more time. 
    * with runs up to 10K rounds:
        - Altough `lr=0.05` converges significantly faster than `lr=0.01` in the first 5K rounds, both runs show obvious signal of plateau by 9K-10K rounds (in both test accuracy curves and test loss curves, therefore not being likely to reach test accuracy higher than 80% in reasonable horizon) 
6. For FedSGD 
    * with runs up to 8000 rounds:
        - `lr=0.05` outperform lr=0.1 obviously when using `wycnn_bn`.
        - using `lr=0.05` with `tf_cnn` is far better than same lr with `wycnn_bn`, so that for further runs, `wycnn_bn` will be abandoned for both FedAvg and FedSGD.
        - Similar to FedAvg, using `tf_cnn` with `lr=0.05` outperforms same model with `lr=0.02`. Moreover, using same model `tf_cnn` and same `lr=0.05`, `t2` performs similarly as `t1` but takes 27% more time.
        - with `tf_cnn`, `lr=0.1` outperforms `lr=0.05` in 8K rounds. *So that it would be interesting to see how lr=0.2 performs in 8K rounds.*
    * with runs up to 30K rounds:
        - using `lr=0.5` the learning curve has crossed 80% around 16K rounds and reached a test accuracy >81% in the end, which suggests that 25K-30K rounds may reveal the entire shape of the learning curve. It would then be interesting to test lr=0.5 over longer rounds.
        - `lr=0.1` achieves little bit lower (almost the same) test accuracy (~83%) with a faster convergence than `lr=0.05`. However `lr=0.1` seems to plateau after 25K rounds whereas `lr=0.05` does not, so that it would be expected that `lr=0.05` may reach 84% test accuracy in next 10K rounds (up to 40K). It would also be interesting to see whether `lr=0.02` can catch up `lr=0.05` in 40K rounds. 40K rounds may take 48 hrs on A.
7. For FedAvg up to 9-10K rounds with `lr={0.05, 0.01}` and FedSGD up to 30K rounds with `lr={0.1, 0.05}`, 
    * although FedAvg converges much faster than FedSGD, **it starts to plateau around 75% test accuracy by 10K rounds and does not seem to improve much further whereas FedSGD reach test accuracy higher than 82% by 30K rounds**. 
    * FedSGD with `lr=0.1` starts to surpass FedAvg with `lr={0.05, 0.01}` after 5K rounds, **which means the tested FedAvg runs have no advantage in speedup over FedSGD for test accuracy of 75% and above**.
    * Moreover, FedAvg with `lr=0.01` does not have very big advantages in convergence speed over FedSGD with `lr=0.1` in the first 5K rounds, **which suggests a probability that FedAvg with an even smaller learning rate such as lr=0.005 may not have speedup gains over FedSGD with lr=0.01 and baseline SGD** since FedAvg using this small learning rate can loose the convergence speed advantage while still being relatively lower in test accuracy.
    * Results for MNIST learning obtained in the previous Exp 2B (non-IID) suggests that combination `{E=5, B=50}` can be outperformed by combinations like `{B=1, B=10}`, `{E=20, B=10}` and `{E=5, B=10}` (the best among these three). 
        - Therefore, one may change to test `{E=5, B=10}` and `{B=1, B=10}` rather than `{E=5, B=50}` which is used currently. 
        - Note that  `{E=5, B=10}` leads to **5x minibatch updates per round** compared to `{E=5, B=50}`, and `{E=1, B=10}` leads to **the same amount of local computation per round** as `{E=5, B=50}`. According to the pervious experience, testing `{E=1, B=10}` can enjoy 1.75x speedup in test time duration whereas `{E=5, B=10}` may use largely the same amount of time as `{E=5, B=50}`. So that one can test `{E=1, B=10}` first.
        - Parameter combination `{B=1, B=10}` does not show improvement over `{B=5, B=50}` with `lr={0.05,0.02}` in 20K rounds, starting to plateau after 10K rounds.


#### 3-B: Per mini-batch update convergence FedAvg vs SGD 
* 300,000 rounds mini-batch updates used in the vanilla FL paper is too many to complete in the allowed time for now. Therefore, one may consider **100,000 mini-batch updates** instead, which is equivalent to
    1. 200 rounds (B=100) or for **100 rounds (B=50) SGD**,
    2. 200 rounds for FedAvg E=5, B=50, C=0.1
    3. 100 rounds for FedAvg E=10, B=50, C=0.1, 50 rounds E=20, B=50, C=0.1, 20 rounds for E=5, B=50, C=1.0
    4. 1000 rounds for FedAvg E=1, B=50, C=0.1
    5. 2000 rounds for FedAvg E=5, B=50, C=0.0
* The learning rates of every setting optimized to produce highest possible test acc over 100,000 mini-batch updates, while sacrificing the convergence speed as least as possible.

#### 3-B1 CNN/IID
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

#### 3-B2 CNN/non-IID
Model        |Method|Data      | T Rnd |Time      | Machine | Frac | E | B | Lr/O       | Optim | Status
-----------  |------|----------| ----  |--------  | -----   |---   |---| - | -------    | ----- | ------
`tf_cnn`,`t1`|SGD   |          | 100   |hrs       | T       |      |   |50 | 0.1        | SGD   | done
`tf_cnn`,`t1`|SGD   |          | 100   |hrs       | T       |      |   |50 | 0.05       | SGD   | done, `selected`
`tf_cnn`,`t1`|SGD   |          | 100   |hrs       | A       |      |   |50 | 0.02       | SGD   | done
`tf_cnn`,`t1`|SGD   |          | 100   |0.57hrs   | A       |      |   |50 | 0.01       | SGD   | done
`tf_cnn`,`t1`|SGD   |          | 100   |hrs       | A       |      |   |50 | 0.005      | SGD   | done
`tf_cnn`,`t1`|FedAVg|non-iid   | 2000  |1.63hrs   | A       | 0.0  |5  |50 | 0.1        | SGD   | done
`tf_cnn`,`t1`|FedAVg|non-iid   | 2000  |hrs       | A       | 0.0  |5  |50 | 0.05       | SGD   | done
`tf_cnn`,`t1`|FedAVg|non-iid   | 2000  |hrs       | A       | 0.0  |5  |50 | 0.02       | SGD   | done, `selected`
`tf_cnn`,`t1`|FedAVg|non-iid   | 2000  |1.59hrs   | A       | 0.0  |5  |50 | 0.01       | SGD   | done
`tf_cnn`,`t1`|FedAVg|non-iid   | 2000  |hrs       | A       | 0.0  |5  |50 | 0.005      | SGD   | done  
`tf_cnn`,`t1`|FedAVg|non-iid   | 1000  |1.09hrs   | A       | 0.1  |1  |50 | 0.1        | SGD   | done, `selected`
`tf_cnn`,`t1`|FedAVg|non-iid   | 1000  |hrs       | A       | 0.1  |1  |50 | 0.05       | SGD   | done
`tf_cnn`,`t1`|FedAVg|non-iid   | 1000  |hrs       | A       | 0.1  |1  |50 | 0.02       | SGD   | done 
`tf_cnn`,`t1`|FedAVg|non-iid   | 1000  |1.0hrs    | A       | 0.1  |1  |50 | 0.2        | SGD   | done
`tf_cnn`,`t1`|FedAVg|non-iid   | 1000  |1.0hrs    | A       | 0.1  |1  |50 | 0.5        | SGD   | done 
`tf_cnn`,`t1`|FedAVg|non-iid   | 200   |~1.0hrs   | A       | 0.1  |5  |50 | 0.05       | SGD   | done, `selected`
`tf_cnn`,`t1`|FedAVg|non-iid   | 100   |0.59hrs   | A       | 0.1  |10 |50 | 0.1        | SGD   | done, `selected`
`tf_cnn`,`t1`|FedAVg|non-iid   | 100   |0.59hrs   | A       | 0.1  |10 |50 | 0.05       | SGD   | done
`tf_cnn`,`t1`|FedAVg|non-iid   | 100   |0.59hrs   | A       | 0.1  |10 |50 | 0.02       | SGD   | done
`tf_cnn`,`t1`|FedAVg|non-iid   | 100   |hrs       | A       | 0.1  |10 |50 | 0.2        | SGD   | done
`tf_cnn`,`t1`|FedAVg|non-iid   | 100   |hrs       | A       | 0.1  |10 |50 | 0.5        | SGD   | done
`tf_cnn`,`t1`|FedAVg|non-iid   | 50    |0.56hrs   | A       | 0.1  |20 |50 | 0.1        | SGD   | done, `selected`
`tf_cnn`,`t1`|FedAVg|non-iid   | 50    |0.56hrs   | A       | 0.1  |20 |50 | 0.05       | SGD   | done
`tf_cnn`,`t1`|FedAVg|non-iid   | 50    |0.56hrs   | A       | 0.1  |20 |50 | 0.02       | SGD   | done
`tf_cnn`,`t1`|FedAVg|non-iid   | 50    |hrs       | A       | 0.1  |20 |50 | 0.2        | SGD   | done
`tf_cnn`,`t1`|FedAVg|non-iid   | 50    |hrs       | A       | 0.1  |20 |50 | 0.5        | SGD   | done 
`tf_cnn`,`t1`|FedAVg|non-iid   | 20    |hrs       | A       | 1.0  |5  |50 | 0.1        | SGD   | done, `selected`
`tf_cnn`,`t1`|FedAVg|non-iid   | 20    |hrs       | A       | 1.0  |5  |50 | 0.05       | SGD   | done
`tf_cnn`,`t1`|FedAVg|non-iid   | 20    |hrs       | A       | 1.0  |5  |50 | 0.02       | SGD   | done 
`tf_cnn`,`t1`|FedAVg|non-iid   | 20    |hrs       | A       | 1.0  |5  |50 | 0.2        | SGD   | done
`tf_cnn`,`t1`|FedAVg|non-iid   | 20    |hrs       | A       | 1.0  |5  |50 | 0.5        | SGD   | done


##### Remarks
1. For E=5, B=50, C=0, lr=0.1 is unstable, lr=0.02 outperforms lr=0.05, so lr=0.01 and lr=0.005 can be tested further
2. For E=1/10/20, B=50, C=0.1, lr=0.1 outperforms lr=0.02 and lr=0.05 outperforms, so lr=0.2 and lr=0.5 can be tested further
3. For E=5, B=50, C=1.0, lr=0.1 outperforms lr=0.02 and lr=0.05 outperforms, so lr=0.2 and lr=0.5 can be tested further

### Experiment 4: Additional FedAvg with very large E and unbalanced non-IID data
#### 4-A FedAvg with large E
* test FedAVg E={100,200,500} B=10 C=0.1 and compare with E=1 over IID/non-IID data, using lr=0.1 (the same as E=1 B=10 C=0.1) to see the effect of very large amount of local computation:

Data | Time  | Machine | C   | E   | B  | Lr   | Optim | Status
-----|-----  |-------- | -   | -   | -  | -    | ----- | ------
IID  |1.64hrs| T       | 0.1 | 50  | 10 | 0.05 | SGD   | done   
IID  |    hrs| A       | 0.1 | 100 | 10 | 0.05 | SGD   | running   
IID  |    hrs| A       | 0.1 | 200 | 10 | 0.05 | SGD   |   
N-IID|       |         | 0.1 | 5   | 10 | 0.05 | SGD   | done   
N-IID|       |         | 0.1 | 25  | 10 | 0.05 | SGD   | done   
N-IID|       |         | 0.1 | 50  | 10 | 0.05 | SGD   | done   
N-IID|3.30hrs|         | 0.1 | 100 | 10 | 0.05 | SGD   | done   
N-IID|       |         | 0.1 | 200 | 10 | 0.05 | SGD   | done   
N-IID|       |         | 0.1 | 400 | 10 | 0.05 | SGD   | done  

#### 4-B Unbalanced Non-IID data
* test FedAVg E=1 B=10 C=0.1 over unbalanced non-IID data, and compare with IID, balanced-non-IID, may use the same lr as FedAvg E=1 B=10 C=0.1 in  balanced-IID 
