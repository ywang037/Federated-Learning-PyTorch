# Work logs and notes on reproducing vanilla FL paper - Part C: Experimental results
*NOTE: all the following results are obtained with WY's models*




### Experiment 1: increase parallism
* The runs using optimized learning rate will be marked as "0.01/o"
* Set 96%, 98% as targets for CNN, 2NN is to avoid the needed rounds from being too large to complete in time.
* R-98 means the number of round where test acc hit 98%, similar for R-XX
* T Rnd means the total number of performed rounds
* Test acc (f,max) means final value and max value 
* The time taken for C=1.0 is formidable, even for IID cases. So, experiment for C=1.0 is cancelled for now.
    * one could consider 200 rounds for C=1.0, E=5, B=10 of non-IID, in order to complete in allowed timeline.

#### CNN/IID
Model        |Method|Data  | Test acc (f,max) |R-98     |T Rnd |Time      | Machine | Frac | E | B | Lr     | Optim | Status
-------------|------|------| --------         |-----    |----- |--------  |-------- | -----|---|---| -----  | ----- | ------
`wycnn_dp`   |FedAVg|iid   | 98.58%,99.23%    |20       |500   |0.42hrs   | A       | 0.0  |5  |10 | 0.1/o  | SGD   | done
`wycnn_dp`   |FedAVg|iid   | 99.36%,99.36%    |6  (3.3x)|100   |0.38hrs   | A       | 0.1  |5  |10 | 0.1/o  | SGD   | done
`wycnn_dp`   |FedAVg|iid   | 99.35%,99.36%    |5  (4.0x)|100   |0.69hrs   | A       | 0.2  |5  |10 | 0.1/o  | SGD   | done
`wycnn_dp`   |FedAVg|iid   | 99.29%,99.32%    |5  (4.0x)|100   |1.54hrs   | A       | 0.5  |5  |10 | 0.1/o  | SGD   | done
`wycnn_dp`   |FedAVg|iid   |                  |         |      |          | A       | 1.0  |5  |10 |        | SGD   | cancelled
`wycnn_dp`   |FedAVg|iid   | 98.19%,98.45%    |88       |100   |0.05hrs   | A       | 0.0  |5  |∞  | 0.1/o  | SGD   | done
`wycnn_dp`   |FedAVg|iid   | 98.36%,98.37%    |67 (1.3x)|100   |0.18hrs   | T       | 0.1  |5  |∞  | 0.2/o  | SGD   | done
`wycnn_dp`   |FedAVg|iid   | 98.66%,98.70%    |54 (1.6x)|100   |0.32hrs   | T       | 0.2  |5  |∞  | 0.2/o  | SGD   | done
`wycnn_dp`   |FedAVg|iid   | 98.48%,98.52%    |55 (1.6x)|100   |0.78hrs   | A       | 0.5  |5  |∞  | 0.2/o  | SGD   | done
`wycnn_dp`   |FedAVg|iid   |                  |         |100   |          | A       | 1.0  |5  |∞  |        | SGD   | cancelled

#### Remarks
1. {E=5, B=∞} cannot reach 99% in allowed time i.e., within 100 rounds.
2. As the fraction paramter C increases from 0.1 to 0.5, there is no significant speed up improvemnt observed, which is in line with the experimental results reported in the vanilla FL paper (referred to as vanilla FL results for short).


#### CNN/non-IID
Model        |Method|Data  | Test acc (f,max) |R-98      |T Rnd |Time      | Machine | Frac | E | B | Lr     | Optim | Status
-------------|------|------| --------         |-----     |----- |--------  |-------- | -----|---|---| -----  | ----- | ------
`wycnn_dp`   |FedAVg|N-iid | 99.27%,99.38%    |416       |1500  |1.4hrs    | A       | 0.0  |5  |10 | 0.04/o | SGD   | done
`wycnn_dp`   |FedAVg|N-iid | 99.21%,99.30%    |75  (5.5x)|1500  |6.25hrs   | A       | 0.1  |5  |10 | 0.04/o | SGD   | done
`wycnn_dp`   |FedAVg|N-iid | 99.11%,99.19%    |40 (10.4x)|250   |1.67hrs   | A       | 0.2  |5  |10 | 0.04/o | SGD   | done
`wycnn_dp`   |FedAVg|N-iid | 98.95%,99.07%    |42  (9.9x)|200   |3.97hrs   | A       | 0.5  |5  |10 | 0.04/o | SGD   | done
`wycnn_dp`   |FedAVg|N-iid |                  |          |      |          | A       | 1.0  |5  |10 |        | SGD   | cancelled
`wycnn_dp`   |FedAVg|N-iid | 98.62%,98.98%    |778       |1500  |1.00hrs   | A       | 0.0  |5  |∞  | 0.02/o | SGD   | done
`wycnn_dp`   |FedAVg|N-iid | 98.92%,99.07%    |323 (2.4x)|1500  |3.08hrs   | A       | 0.1  |5  |∞  | 0.04/o | SGD   | done
`wycnn_dp`   |FedAVg|N-iid | 98.99%,99.08%    |319 (2.4x)|1500  |5.54rs    | A       | 0.2  |5  |∞  | 0.04/o | SGD   | done
`wycnn_dp`   |FedAVg|N-iid | 98.36%,98.36%    |318 (2.5x)|400   |3.40hrs   | A       | 0.5  |5  |∞  | 0.04/o | SGD   | done
`wycnn_dp`   |FedAVg|N-iid |                  |          |      |          | A       | 1.0  |5  |∞  |        | SGD   | cancelled

##### Remarks
1. Similar to IID cases, {E=5, B=∞} cannot reach 99% in allowed time, i.e., within 400 rounds.
2. Similar to IID cases, increasing the fraction paramter C from 0.1 to 0.5 does not necessarily improve or improves little the speed up, which is in line with the vanilla FL results.



### Experiment 2: increase local computation
* The fraction number is fixed at C=0.1.
* The runs using optimized learning rate will be marked as "0.01/o".
* R-98 means the number of round where test acc hit 98%, similar for R-XX.
* T Rnd means the total number of performed rounds.
* Test acc (f,max) means final value and max value.
* Status marked as "bm" indicate the benchmark runs, and "bm/d" indicate discarded benchmark runs.

#### CNN/IID
Model        |Method|Data  | Test acc (f,max) |R-98       |R-99  |T Rnd |Time      | Machine | Frac | E | B | Lr     | Optim | Status
-------------|------|------| --------         |-----      |----- |----  |--------  | -----   |---   |---| - | -----  | ----- | ------
`wycnn_dp`   |FedSGD|iid   | 98.94%,99.01%    | 200       | 573  |600   |0.40hrs   | T       | 0.1  |1  |∞  | 0.2/o  | SGD   | bm/d
`wycnn_dp`   |FedSGD|iid   | 16.34%,99.08%    | 210       | 603  |1000  |0.66hrs   | T       | 0.1  |1  |∞  | 0.15/o | SGD   | bm/d
`wycnn_dp`   |FedSGD|iid   | 99.15%,99.09%    | 230       | 614  |1500  |1.02hrs   | A       | 0.1  |1  |∞  | 0.1/o  | SGD   | done
`wycnn_dp`   |FedAVg|iid   | 99.02%,99.09%    | 57  (x.xx)| 253  |600   |1.10hrs   | T       | 0.1  |5  |∞  | 0.1/o  | SGD   | done 
`wycnn_dp`   |FedAVg|iid   | 99.16%,99.12%    | 49  (x.xx)| 163  |600   |1.16hrs   | A       | 0.1  |5  |∞  | 0.15/o | SGD   | bm 
`wycnn_dp`   |FedAVg|iid   | 99.26%,99.41%    | 19  (x.xx)| 55   |600   |0.41hrs   | T       | 0.1  |1  |50 | 0.2/o  | SGD   | done
`wycnn_dp`   |FedAVg|iid   | 99.18%,99.10%    | 20  (x.xx)| 253  |600   |3.69hrs   | T       | 0.1  |20 |∞  | 0.15/o | SGD   | done
`wycnn_dp`   |FedAVg|iid   | 99.10%,98.98%    | 24  (x.xx)| 164  |600   |3.69hrs   | T       | 0.1  |20 |∞  | 0.15/o | SGD   | re-done
`wycnn_dp`   |FedAVg|iid   | 99.19%,97.73%    | 17  (x.xx)| 121  |600   |3.89hrs   | A       | 0.1  |20 |∞  | 0.2/o  | SGD   | bm/d
`wycnn_dp`   |FedAVg|iid   | 99.16%,99.11%    | 21  (x.xx)| 122  |600   |3.86hrs   | A       | 0.1  |20 |∞  | 0.1/o  | SGD   | bm 
`wycnn_dp`   |FedAVg|iid   | 99.31%,99.41%    | 10  (x.xx)| 32   |600   |0.64hrs   | T       | 0.1  |1  |10 | 0.1/o  | SGD   | done
`wycnn_dp`   |FedAVg|iid   | 99.28%,99.21%    | 8   (x.xx)| 39   |600   |1.24hrs   | T       | 0.1  |5  |50 | 0.15/o | SGD   | done
`wycnn_dp`   |FedAVg|iid   | 99.38%,99.29%    | 11  (x.xx)| 50   |600   |1.55hrs   | A       | 0.1  |5  |50 | 0.1/o  | SGD   | bm
`wycnn_dp`   |FedAVg|iid   | 99.28%,99.23%    | 8   (x.xx)| 37   |600   |4.23hrs   | T       | 0.1  |20 |50 | 0.15/o | SGD   | done
`wycnn_dp`   |FedAVg|iid   | 99.26%,99.18%    | 10  (x.xx)| 46   |600   |4.12hrs   | T       | 0.1  |20 |50 | 0.1/o  | SGD   | bm
`wycnn_dp`   |FedAVg|iid   | 99.41%,99.38%    | 5   (x.xx)| 16   |600   |2.29hrs   | T       | 0.1  |5  |10 | 0.15/o | SGD   | done
`wycnn_dp`   |FedAVg|iid   | 99.38%,99.31%    | 5   (x.xx)| 18   |600   |7.48hrs   | T       | 0.1  |20 |10 | 0.1/o  | SGD   | done

##### Remarks
1. For {E=1, B=inf}, lr=0.2 and lr=0.15 produce similar results in 600 rounds; lr=0.15 become unstable at last few rounds near 1000, so would the even larger lr=0.2. It seems that lr=0.1 might be more reasonable learning rate.
2. For {E=1, B=10}, lr=0.2 and 0.15 is unstable in real test run, try {0.1, 0.07} instead. The other parameter combinations can follow similar approach for dealing instability under large lr.
3. For {E=5, B=50}, lr=0.2 is unstable in real test run after 297 rounds, try {0.15, 0.1, 0.07} instead.
    * Compared with lr=0.1, lr=0.15 leads to slightly faster convergence at the expense of a little bit of test acc. 
4. For {E=20, B=inf}, lr=0.2 accelearates the speed for reaching 99% test acc, but also become unstable around 422 rounds, and leads to lower error floor in the end. 
5. For {E=20, B=50}, lr=0.15 leads to unstable pike around 800 rounds; so bm tries lr=0.1.
5. It can observed that the more local computation, the quicker the convergence (or the higher test acc that can be achived within the identical number or rounds). 
6. Larger learning rates in general lead to quicker convergence or higher test acc in a given number of rounds, at the expense of being prone to instability. 
7. From the entries of R-98 column, it can be seen that increasing the local computation can lead to more speed up in general.

#### CNN/non-IID
Model        |Method|Data  | Test acc (f,max) |R-98       |R-99  |T Rnd |Time      | Machine | Frac | E | B | Lr     | Optim | Status
-------------|------|------| --------         |-----      |----  |----  |--------  | -----   |---   |---| - | -----  | ----- | ------
`wycnn_dp`   |FedSGD|N-iid | 98.75%,98.83%    | 621       |      |1500  |1.12hrs   | T       | 0.1  |1  |∞  | 0.04/o | SGD   | done
`wycnn_dp`   |FedAVg|N-iid | 97.65%,98.15%    | 567 (    )|      |600   |1.17hrs   | A       | 0.1  |5  |∞  | 0.02/o | SGD   | bm/d
`wycnn_dp`   |FedAVg|N-iid | 98.73%,98.81%    | 264 (2.4x)|      |600   |1.25hrs   | A       | 0.1  |5  |∞  | 0.04/o | SGD   | done
`wycnn_dp`   |FedAVg|N-iid | 98.54%,98.97%    | 152 (4.1x)|      |600   |0.65hrs   | A       | 0.1  |1  |50 | 0.04/o | SGD   | done
`wycnn_dp`   |FedAVg|N-iid | 98.79%,98.93%    | 157 (4.0x)|      |600   |3.99hrs   | A       | 0.1  |20 |∞  | 0.04/o | SGD   | done
`wycnn_dp`   |FedAVg|N-iid | 99.17%,99.35%    | 79  (7.9x)| 204  |600   |0.70hrs   | T       | 0.1  |1  |10 | 0.04/o | SGD   | done
`wycnn_dp`   |FedAVg|N-iid | 98.91%,99.06%    | 109 (5.7x)| 344  |600   |1.61hrs   | A       | 0.1  |5  |50 | 0.04/o | SGD   | bm/d
`wycnn_dp`   |FedAVg|N-iid | 99.07%,99.15%    | 87  (7.1x)| 340  |600   |1.27hrs   | A       | 0.1  |5  |50 | 0.04/o | SGD   | done
`wycnn_dp`   |FedAVg|N-iid | 99.04%,99.16%    | 73  (8.5x)| 351  |600   |4.12hrs   | T       | 0.1  |20 |50 | 0.04/o | SGD   | done
`wycnn_dp`   |FedAVg|N-iid | 99.22%,99.30%    | 42 (14.8x)| 145  |600   |2.58hrs   | A       | 0.1  |5  |10 | 0.04/o | SGD   | done
`wycnn_dp`   |FedAVg|N-iid | 99.15%,99.23%    | 55 (11.3x)| 175  |600   |7.36hrs   | T       | 0.1  |20 |10 | 0.04/o | SGD   | done
`wycnn_dp`   |FedAVg|N-iid |      %,     %    |    (    x)| xxx  |600   |7.36hrs   | T       | 0.1  |20 |10 | 0.04/o | SGD   | to bm

##### Remarks
1. It can observed that, in general, the more local computation, the quicker the convergence (or the higher test acc that can be achived within the identical number or rounds). This is why parameter combinations have fewer local computations cannot reach 99% in same number of rounds (600 rounds).
2. From the entries of R-98 column, it can be seen that increasing the local computation can lead to more speed up in general.


#### CNN/IID (selected edition)
Model        |Method|Data  | Test acc (f,max) |R-98       |R-99        |T Rnd |Time      | Machine | Frac | E | B | Lr     | Optim | Status
-------------|------|------| --------         |-----      |-----       |----  |--------  | -----   |---   |---| - | -----  | ----- | ------
`wycnn_dp`   |FedSGD|iid   | 99.15%,99.09%    | 230       | 614        |1500  |1.02hrs   | A       | 0.1  |1  |∞  | 0.1/o  | SGD   | done
`wycnn_dp`   |FedAVg|iid   | 99.16%,99.12%    | 49  (4.7x)| 163  (1.4x)|600   |1.16hrs   | A       | 0.1  |5  |∞  | 0.15/o | SGD   | bm 
`wycnn_dp`   |FedAVg|iid   | 99.26%,99.41%    | 19 (12.1x)| 55   (4.2x)|600   |0.41hrs   | T       | 0.1  |1  |50 | 0.2/o  | SGD   | done
`wycnn_dp`   |FedAVg|iid   | 99.16%,99.11%    | 21 (11.0x)| 122  (1.9x)|600   |3.86hrs   | A       | 0.1  |20 |∞  | 0.1/o  | SGD   | bm 
`wycnn_dp`   |FedAVg|iid   | 99.31%,99.41%    | 10 (23.0x)| 32   (7.2x)|600   |0.64hrs   | T       | 0.1  |1  |10 | 0.1/o  | SGD   | done
`wycnn_dp`   |FedAVg|iid   | 99.28%,99.21%    | 8  (28.8x)| 39   (5.9x)|600   |1.24hrs   | T       | 0.1  |5  |50 | 0.15/o | SGD   | done
`wycnn_dp`   |FedAVg|iid   | 99.28%,99.23%    | 8  (28.8x)| 37   (6.2x)|600   |4.23hrs   | T       | 0.1  |20 |50 | 0.15/o | SGD   | done
`wycnn_dp`   |FedAVg|iid   | 99.41%,99.38%    | 5  (46.0x)| 16  (14.4x)|600   |2.29hrs   | T       | 0.1  |5  |10 | 0.15/o | SGD   | done
`wycnn_dp`   |FedAVg|iid   | 99.38%,99.31%    | 5  (46.0x)| 18  (12.8x)|600   |7.48hrs   | T       | 0.1  |20 |10 | 0.1/o  | SGD   | done

##### Remarks
1. Learning curve of {E=5, B=50} is very noisy and unstable, may need a re-run

#### CNN/non-IID (selected edition)
Model        |Method|Data  | Test acc (f,max) |R-98       |R-99  |T Rnd |Time      | Machine | Frac | E | B | Lr     | Optim | Status
-------------|------|------| --------         |-----      |----  |----  |--------  | -----   |---   |---| - | -----  | ----- | ------
`wycnn_dp`   |FedSGD|N-iid | 98.75%,98.83%    | 621       |      |1500  |1.12hrs   | T       | 0.1  |1  |∞  | 0.04/o | SGD   | done
`wycnn_dp`   |FedAVg|N-iid | 98.73%,98.81%    | 264 (2.4x)|      |600   |1.25hrs   | A       | 0.1  |5  |∞  | 0.04/o | SGD   | done
`wycnn_dp`   |FedAVg|N-iid | 98.54%,98.97%    | 152 (4.1x)|      |600   |0.65hrs   | A       | 0.1  |1  |50 | 0.04/o | SGD   | done
`wycnn_dp`   |FedAVg|N-iid | 98.79%,98.93%    | 157 (4.0x)|      |600   |3.99hrs   | A       | 0.1  |20 |∞  | 0.04/o | SGD   | done
`wycnn_dp`   |FedAVg|N-iid | 99.17%,99.35%    | 79  (7.9x)| 204  |600   |0.70hrs   | T       | 0.1  |1  |10 | 0.04/o | SGD   | done
`wycnn_dp`   |FedAVg|N-iid | 99.07%,99.15%    | 87  (7.1x)| 340  |600   |1.27hrs   | A       | 0.1  |5  |50 | 0.04/o | SGD   | done
`wycnn_dp`   |FedAVg|N-iid | 99.04%,99.16%    | 73  (8.5x)| 351  |600   |4.12hrs   | T       | 0.1  |20 |50 | 0.04/o | SGD   | done
`wycnn_dp`   |FedAVg|N-iid | 99.22%,99.30%    | 42 (14.8x)| 145  |600   |2.58hrs   | A       | 0.1  |5  |10 | 0.04/o | SGD   | done
`wycnn_dp`   |FedAVg|N-iid | 99.15%,99.23%    | 55 (11.3x)| 175  |600   |7.36hrs   | T       | 0.1  |20 |10 | 0.04/o | SGD   | done



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

Model       |Method|Data Augmentation                 | Test acc (max) | Epoch |Time      | Machine | B  | Lr/O   | Optim | Decay |  Status
-------     |------|------------------------------    |--------------- | ----  |--------  | -----   | -- | -----  |------ | ----- | ------
`wycnn_dp`  |SGD   |t0: default                       | 73.8%          | 200   |hrs       | T       |100 | 0.01   | SGD   |       | benchmark
`wycnn_dp`  |SGD   |t1: mean, std, crop, flip         | 80.0%          | 200   |1.05hrs   | T       |100 | 0.01   | SGD   |       | done
`wycnn_dp`  |SGD   |t2: mean, std, crop, flip, color  | 79.4%          | 200   |1.61hrs   | T       |100 | 0.01   | SGD   |       | done
`wycnn_dp`  |SGD   |t1: mean, std, crop, flip         | 82.3%          | 400   |2.09hrs   | T       |100 | 0.01   | SGD   |       | done
`wycnn_bn`  |SGD   |t1: mean, std, crop, flip         | 84.5%          | 200   |1.10hrs   | T       |100 | 0.01   | SGD   |       | done
`wycnn_bn`  |SGD   |t1: mean, std, crop, flip         | 85.0%          | 400   |2.15hrs   | T       |100 | 0.01   | SGD   |       | done
`wycnn_bn`  |SGD   |t2: mean, std, crop, flip, color  | 85.2%          | 400   |3.55hrs   | A       |100 | 0.01   | SGD   |       | done
`wycnn_bndp`|SGD   |t1: mean, std, crop, flip         | 83.3%          | 400   |2.17hrs   | T       |100 | 0.01   | SGD   |       | done
`tf_cnn`    |SGD   |t1: mean, std, crop, flip         | 84.2%          | 400   |1.85hrs   | T       |100 | 0.01   | SGD   |       | done, `selected`
`tf_cnn`    |SGD   |t2: mean, std, crop, flip, color  | 84.2%          | 400   |3.54hrs   | A       |100 | 0.01   | SGD   |       | done


#### 3-A: Speed up of convergence agaisnt comm. round, FedAvg vs FedSGD vs SGD
* Fraction of users is fixed at C=0.1, FedSGD and FedAVg use fixed E=5, and FedAvg use fixed B=50
* One cannot afford to perform that many rounds for FedSGD, a reasonable approach is to let FedAvg and FedSGD perform identical rounds of learning, as what are performed in the previous two experiments, e.g., 100-200 rounds.
    * Considering either *torch cnn* and *tf cnn* overfits in less than 100 epochs of SGD, one may perform 100 rounds FedAvg/FedSGD
    * Reduced rounds of experiment is a make-do method, it is acceptable for the time being as long as the experiment result can reflect the same fundamental conclusion drawn in the vaniila FL paper
* Decay means the learning-rate decay. No decay is tuned and applied

#### 3-A1 CNN/IID
Model |Method|Data  | T Rnd |Time      | Machine | Frac | E | B | Lr/O      |Decay  | Optim | Status
------|------|------| ----  |--------  | -----   |---   |---| - | -----     |------ | ----- | ------
CNN   |SGD   |iid   | xxxx  |hrs       | T       |      |   |100| 0.01@dp   |       | SGD   | run on A
CNN   |FedSGD|iid   | 8000  |~8hrs     | A       | 0.1  |1  |∞  | 0.1@dp    |       | SGD   | done
CNN   |FedAVg|iid   | 4000  |9.18hrs   | T       | 0.1  |5  |50 | 0.03@dp   |       | SGD   | done

##### Remarks
0. In the above table, the learning rate of each run is selected from all conducted grid searches so as to produce highest possible test acc within allowed time.
1. Experiments with 2000 rounds show that:
    * For SGD, for lr in {2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 1.5e-2, 2e-2, 3e-2}, larger lr converge to 72% test acc quicker but tends to overfit earlier and needs the "ealier stopping". ***Among all lr, lr=0.01 reaches 72% around 45 rounds and achieves a maximum test acc about 74%***. Futher tests using learning-rate decay of 0.99 does not show any improvement for SGD with lr=0.01 in 200 rounds. The above observations suggest that even smaller lr like {5e-5, 1e-4, ..., 1e-3} may reach a higer test acc in long-term runs over, say, 10000 rounds.
    * For FedAvg, larger lr in {0.05,0.07,0.1} converge quicker before 200 rounds, ***however, lr=0.05 outperforms and converges to 0.72 quicker afterwards***.
    * For FedSGD, coarse searches in {0.05, 0.1, 0.2} find that lr=0.2 keeps increasing in test acc in 4000 rounds (3.8 hrs), and is better than smaller lr=0.1,0.05. So, next searches could be within {0.1, 0.2, 0.5} over 8000 rounds.
2. Tests over 4000/8000 rounds show that:
    * For FedAvg, tests of lr={0.005, 0.01, 0.03, 0.05} in 4000 rounds show that ***lr=0.03 reach a higher test acc***. Both lr=0.05 and lr=0.01 converge in slower speed and to lower test acc, lr=0.05 converges a little bit quicker than lr=0.03 but it converges to a similar test acc as lr=0.01. None of {0.005,0.01,0.05} reach 74%. 
    * For FedSGD, test of lr={0.1, 0.2, 0.5} for 8000 rounds show that the larger lr the quicker the convergence, but ***lr=0.1 reach the highest test acc among these three lr***, and the related test loss is just about to rise again (just not to overfit), which suggests that in 8000 rounds, even smaller lr like 0.05, 0.03, and 0.02 may not achieve same test acc and also converge slower compared with lr=0.1. 
3. If the target is to compare speedup for a specific test acc target, then one may not need to care too much about how to reach a higest final test acc for each algorithm. 
    * Instead, one may wish to tune the quickest fashion for the convergence of each algorithm, and then do the comparison of speedup.
    * Limited computational power and project timeline does not allow for
        - Finer searches of best learning rate (and decay scheme) for each algorithm.
        - Long-term runs (e.g., 200,000 rounds) of SGD in order to obtain the highest possible test acc is not fesible for the time being.
    * The convergence objective can be set to 0.68, 0.7, 0.72, 0.74

#### 3-A2 CNN/non-IID
Model      |Method|Data    | T Rnd |Time      | Machine | Frac | E | B | Lr/O      |Decay  | Optim | Augmentation   |Status
---------- |------|------- | ----  |--------  | -----   |---   |---| - | -----     |------ | ----- | -----------    |------
`tf_cnn`   |SGD   |iid     | 400   |hrs       | A       |      |   |100| 0.05      |       | SGD   | `t1` transform | done, `selected`
`tf_cnn`   |FedSGD|non-iid | 8000  | 8.3hrs   | A       | 0.1  |1  |∞  | 0.05      |       | SGD   | `t1` transform | done, `selected`
`tf_cnn`   |FedAvg|non-iid | 4000  |13.6hrs   | A       | 0.1  |5  |50 | 0.05      |       | SGD   | `t1` transform | done, `selected`

##### Remarks
1. Further baseline test runs show that the alternative transform leads to better performance in both convergence rate and test acc when using the same lr as standard transform. However, in this series of tests, the standard transform is still used for the consistency.
2. For these non-IID tests regarding CIFAR10, the baseline SGD is the same as IID tests.
3. For SGD with runs up to 400 epochs:
    - `lr=0.1` has quickest convergence in first 50 epochs, `lr=0.05` has the quickest convergence in first 250 epochs, `lr=0.01` has highest test accuracy in 400 epochs.
4. For FedAvg with runs up to 4000 rounds:
    - It has been observed that in these non-IID tests, the learning curves oscillates a lot more than those in IID cases, for all the lr tried.
    - Larger lr in {0.005, 0.01, 0.02, 0.05, 0.1} performs better up to 2K rounds.
    - Test run with `wycnn_bn` and `t1` outperform `wycnn_dp` and `t1` in the first 2K rounds but then become similarly bad. This run also seems to severely overfit after 1K rounds.
    - **Unlike the case of SGD** (where data is not non-IID), in this non-IID setting, performance of FedAvg using `wycnn_bn` and data augmentation `t1` is much worsen than using `tf_cnn` with `t1`.
    - Using `tf_cnn` with `lr=0.05` and data augmentation produces best results where `t2` performs similarly as `t1` up to 4K rounds, however `t2` takes about 47% more time. 
5. For FedSGD with runs up to 8000 rounds:
    - `lr=0.05` outperform lr=0.1 obviously when using `wycnn_bn` and `t1`.
    - using `lr=0.05` with `tf_cnn` and `t1` is far better than same lr with `wycnn_bn` and `t1`, so that for further runs, `wycnn_bn` will be abandoned for both FedAvg and FedSGD.
    - Similar to FedAvg, using `tf_cnn` with `lr=0.05` and `t1` outperforms same model with `lr=0.02` and `t1`. Moreover, using same model `tf_cnn` and same `lr=0.05`, `t2` performs similarly as `t1` but takes 27% more time.
6. As runs of FedAvg and FedSGD do not achieve test accuracy higher than 78% in the allowed time (4K and 8K rounds, respectively), the target test accuracy to benchmark is chosen as {72%, 74%, 76%}.



#### 3-B: Per mini-batch update convergence FedAvg vs SGD 
* For SGD, batch size is fixed at B=100, so number of mini-batch updates is $500R$ since N=50,000 so mini-batch update per round is N/B=500.
* For FedAvg, since E, B are fixed, so number of mini-batch updates per participant is $n=R\times \frac{NE}{100B}=50R$, and the total number of mini-batch updates conducted by all participant clients is $50R/C$, C is the fraction parameter.
* 300,000 rounds mini-batch updates used in the vanilla FL paper is too many to complete in the allowed time for now. Therefore, one may consider **100,000 mini-batch updates** instead, which is equivalent to
    1. 200 rounds (B=100) or for **100 rounds (B=50) SGD**,
    2. 200 rounds for FedAvg E=5, B=50, C=0.1
    3. 100 rounds for FedAvg E=10, B=50, C=0.1, 50 rounds E=20, B=50, C=0.1, 20 rounds for E=5, B=50, C=1.0
    4. 1000 rounds for FedAvg E=1, B=50, C=0.1
    5. 2000 rounds for FedAvg E=5, B=50, C=0.0
* The learning rates of every setting optimized to produce highest possible test acc over 100,000 mini-batch updates, while sacrificing the convergence speed as least as possible.


#### 3-B1 CNN/IID
Model        |Method|Data  | T Rnd |Updata per R|Time      | Machine | Frac | E | B | Lr/O    | Optim | Status
-------------|------|------| ----  |--------    |--------  | -----   |---   |---| - | -----   | ----- | ------
`wycnn_dp`   |SGD   |iid   | 100   |1000        |hrs       | T       |      |   |50 | 0.0032  | SGD   | done
`wycnn_dp`   |FedAVg|iid   | 2000  |50          |hrs       | T       | 0.0  |5  |50 | 0.02    | SGD   | done
`wycnn_dp`   |FedAVg|iid   | 1000  |100         |hrs       | T       | 0.1  |1  |50 | 0.1     | SGD   | done
`wycnn_dp`   |FedAVg|iid   | 200   |500         |hrs       | T       | 0.1  |5  |50 | 0.1     | SGD   | done
`wycnn_dp`   |FedAVg|iid   | 100   |1000        |hrs       | T       | 0.1  |10 |50 | 0.2     | SGD   | done
`wycnn_dp`   |FedAVg|iid   | 50    |2000        |hrs       | T       | 0.1  |20 |50 | 0.2     | SGD   | done
`wycnn_dp`   |FedAVg|iid   | 20    |5000        |hrs       | T       | 1.0  |5  |50 | 0.32    | SGD   | done

##### Remarks
0. In the above table, the learning rate of each run is selected from all conducted grid searches so as to produce highest possible test acc within 100,000 mini-batch update.
1. For SGD with B=50, among lr={0.002, 0.0032, 0.0046, 0.005, 0.01, 0.02}, lr=0.1 converge to 73% test acc quickest (within 40 rounds), but smaller lrs reach a slightly higher maximum test acc. One can choose to show 
    * lr=0.01 with early stopping @40 rounds; or
    * lr=0.005 with early stopping @80 rounds; or
    * lr=0.0032 with 100-round full run
2. For FedAvg E=5, B=50, C=0.0, among lr={0.01, 0.02, 0.032, 0.05, 0.1, 0.2}, lr=0.02 and lr=0.32 outperform the others, one can choose to show between these two runs.
3. For FedAvg E=1, B=50, C=0.1, lr=0.1 works better than 0.05 and 0.2 within 1000 rounds.
4. For FedAvg E=5, B=50, C=0.1, lr=0.1 works better than 0.05 within 200 rounds (number of rounds that this parameter combination needs to do for 100,000 mini-batch updates for this experiment).
5. For FedAvg E=10, B=50, C=0.1, tests over 100 rounds show that lr=0.2 converges achieve highest test acc and converges quickest among lr={0.001, 0.01, 0.1, 0.2, 0.32, 0.5, 0.7}.
6. For FedAvg E=20, B=50, C=0.1, tests over 50 rounds show that lr=0.2 converges achieve highest test acc and converges quickest among lr={0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 0.7}.
7. For FedAvg E=5, B=50, C=1.0, larger lr in {0.001, 0.01, 0.1, 0.2, 0.32} achieve higher test acc in 20 round and also converges quicker, lr=0.5 converges even slower than lr=0.1 and become unstable after 10 rounds.

#### 3-B2 CNN/non-IID
Model        |Method|Data      | T Rnd |Time      | Machine | Frac | E | B | Lr/O       | Optim | Status
-----------  |------|----------| ----  |--------  | -----   |---   |---| - | -------    | ----- | ------
`tf_cnn`,`t1`|SGD   |          | 100   |hrs       | A       |      |   |50 | 0.05       | SGD   | done, `selected`
`tf_cnn`,`t1`|FedAVg|non-iid   | 2000  |hrs       | A       | 0.0  |5  |50 | 0.02       | SGD   | done, `selected`
`tf_cnn`,`t1`|FedAVg|non-iid   | 1000  |1.09hrs   | A       | 0.1  |1  |50 | 0.1        | SGD   | done, `selected`
`tf_cnn`,`t1`|FedAVg|non-iid   | 200   |~1.0hrs   | A       | 0.1  |5  |50 | 0.05       | SGD   | done, `selected`
`tf_cnn`,`t1`|FedAVg|non-iid   | 100   |0.59hrs   | A       | 0.1  |10 |50 | 0.1        | SGD   | done, `selected`
`tf_cnn`,`t1`|FedAVg|non-iid   | 50    |0.56hrs   | A       | 0.1  |20 |50 | 0.1        | SGD   | done, `selected`
`tf_cnn`,`t1`|FedAVg|non-iid   | 20    |hrs       | A       | 1.0  |5  |50 | 0.1        | SGD   | done, `selected`


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
N-IID|       | T       | 0.1 | 50  | 10 | 0.05 | SGD   | done   
N-IID|3.30hrs| T       | 0.1 | 100 | 10 | 0.05 | SGD   | done   
N-IID|       | A       | 0.1 | 200 | 10 | 0.05 | SGD   | running   

#### 4-B Unbalanced Non-IID data
* test FedAVg E=1 B=10 C=0.1 over unbalanced non-IID data, and compare with IID, balanced-non-IID, may use the same lr as FedAvg E=1 B=10 C=0.1 in  balanced-IID 


