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
* Status marked as "bm" indicate the benchmark runs, and "bm/d" indicate discarded benchmark runs.

##### CNN/IID
Model |Method|Data  | Test acc (f,max) |R-98       |R-99  |T Rnd |Time      | Machine | Frac | E | B | Lr     | Optim | Status
------|------|------| --------         |-----      |----- |----  |--------  | -----   |---   |---| - | -----  | ----- | ------
CNN   |FedSGD|iid   | 98.94%,99.01%    | 200       | 573  |600   |0.40hrs   | T       | 0.1  |1  |∞  | 0.2/o  | SGD   | bm/d
CNN   |FedSGD|iid   | 16.34%,99.08%    | 210       | 603  |1000  |0.66hrs   | T       | 0.1  |1  |∞  | 0.15/o | SGD   | bm/d
CNN   |FedSGD|iid   | 99.15%,99.09%    | 230       | 614  |1500  |1.02hrs   | A       | 0.1  |1  |∞  | 0.1/o  | SGD   | done
CNN   |FedAVg|iid   | 99.02%,99.09%    | 57  (x.xx)| 253  |600   |1.10hrs   | T       | 0.1  |5  |∞  | 0.1/o  | SGD   | done 
CNN   |FedAVg|iid   | 99.16%,99.12%    | 49  (x.xx)| 163  |600   |1.16hrs   | A       | 0.1  |5  |∞  | 0.15/o | SGD   | bm 
CNN   |FedAVg|iid   | 99.26%,99.41%    | 19  (x.xx)| 55   |600   |0.41hrs   | T       | 0.1  |1  |50 | 0.2/o  | SGD   | done
CNN   |FedAVg|iid   | 99.18%,99.10%    | 20  (x.xx)| 253  |600   |3.69hrs   | T       | 0.1  |20 |∞  | 0.15/o | SGD   | done
CNN   |FedAVg|iid   | 99.10%,98.98%    | 24  (x.xx)| 164  |600   |3.69hrs   | T       | 0.1  |20 |∞  | 0.15/o | SGD   | re-done
CNN   |FedAVg|iid   | 99.19%,97.73%    | 17  (x.xx)| 121  |600   |3.89hrs   | A       | 0.1  |20 |∞  | 0.2/o  | SGD   | bm/d
CNN   |FedAVg|iid   | 99.16%,99.11%    | 21  (x.xx)| 122  |600   |3.86hrs   | A       | 0.1  |20 |∞  | 0.1/o  | SGD   | bm 
CNN   |FedAVg|iid   | 99.31%,99.41%    | 10  (x.xx)| 32   |600   |0.64hrs   | T       | 0.1  |1  |10 | 0.1/o  | SGD   | done
CNN   |FedAVg|iid   | 99.28%,99.21%    | 8   (x.xx)| 39   |600   |1.24hrs   | T       | 0.1  |5  |50 | 0.15/o | SGD   | done
CNN   |FedAVg|iid   | 99.38%,99.29%    | 11  (x.xx)| 50   |600   |1.55hrs   | A       | 0.1  |5  |50 | 0.1/o  | SGD   | bm
CNN   |FedAVg|iid   | 99.28%,99.23%    | 8   (x.xx)| 37   |600   |4.23hrs   | T       | 0.1  |20 |50 | 0.15/o | SGD   | done
CNN   |FedAVg|iid   | 99.26%,99.18%    | 10  (x.xx)| 46   |600   |4.12hrs   | T       | 0.1  |20 |50 | 0.1/o  | SGD   | bm
CNN   |FedAVg|iid   | 99.41%,99.38%    | 5   (x.xx)| 16   |600   |2.29hrs   | T       | 0.1  |5  |10 | 0.15/o | SGD   | done
CNN   |FedAVg|iid   | 99.38%,99.31%    | 5   (x.xx)| 18   |600   |7.48hrs   | T       | 0.1  |20 |10 | 0.1/o  | SGD   | done

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

##### CNN/non-IID
Model |Method|Data  | Test acc (f,max) |R-98       |R-99  |T Rnd |Time      | Machine | Frac | E | B | Lr     | Optim | Status
------|------|------| --------         |-----      |----  |----  |--------  | -----   |---   |---| - | -----  | ----- | ------
CNN   |FedSGD|N-iid | 98.75%,98.83%    | 621       |      |1500  |1.12hrs   | T       | 0.1  |1  |∞  | 0.04/o | SGD   | done
CNN   |FedAVg|N-iid | 97.65%,98.15%    | 567 (    )|      |600   |1.17hrs   | A       | 0.1  |5  |∞  | 0.02/o | SGD   | bm/d
CNN   |FedAVg|N-iid | 98.73%,98.81%    | 264 (2.4x)|      |600   |1.25hrs   | A       | 0.1  |5  |∞  | 0.04/o | SGD   | done
CNN   |FedAVg|N-iid | 98.54%,98.97%    | 152 (4.1x)|      |600   |0.65hrs   | A       | 0.1  |1  |50 | 0.04/o | SGD   | done
CNN   |FedAVg|N-iid | 98.79%,98.93%    | 157 (4.0x)|      |600   |3.99hrs   | A       | 0.1  |20 |∞  | 0.04/o | SGD   | done
CNN   |FedAVg|N-iid | 99.17%,99.35%    | 79  (7.9x)| 204  |600   |0.70hrs   | T       | 0.1  |1  |10 | 0.04/o | SGD   | done
CNN   |FedAVg|N-iid | 98.91%,99.06%    | 109 (5.7x)| 344  |600   |1.61hrs   | A       | 0.1  |5  |50 | 0.04/o | SGD   | bm/d
CNN   |FedAVg|N-iid | 99.07%,99.15%    | 87  (7.1x)| 340  |600   |1.27hrs   | A       | 0.1  |5  |50 | 0.04/o | SGD   | done
CNN   |FedAVg|N-iid | 99.04%,99.16%    | 73  (8.5x)| 351  |600   |4.12hrs   | T       | 0.1  |20 |50 | 0.04/o | SGD   | done
CNN   |FedAVg|N-iid | 99.22%,99.30%    | 42 (14.8x)| 145  |600   |2.58hrs   | A       | 0.1  |5  |10 | 0.04/o | SGD   | done
CNN   |FedAVg|N-iid | 99.15%,99.23%    | 55 (11.3x)| 175  |600   |7.36hrs   | T       | 0.1  |20 |10 | 0.04/o | SGD   | done
CNN   |FedAVg|N-iid |      %,     %    |    (    x)| xxx  |600   |7.36hrs   | T       | 0.1  |20 |10 | 0.04/o | SGD   | to bm

##### Remarks
1. It can observed that, in general, the more local computation, the quicker the convergence (or the higher test acc that can be achived within the identical number or rounds). This is why parameter combinations have fewer local computations cannot reach 99% in same number of rounds (600 rounds).
2. From the entries of R-98 column, it can be seen that increasing the local computation can lead to more speed up in general.


##### CNN/IID (selected edition)
Model |Method|Data  | Test acc (f,max) |R-98       |R-99        |T Rnd |Time      | Machine | Frac | E | B | Lr     | Optim | Status
------|------|------| --------         |-----      |-----       |----  |--------  | -----   |---   |---| - | -----  | ----- | ------
CNN   |FedSGD|iid   | 99.15%,99.09%    | 230       | 614        |1500  |1.02hrs   | A       | 0.1  |1  |∞  | 0.1/o  | SGD   | done
CNN   |FedAVg|iid   | 99.16%,99.12%    | 49  (4.7x)| 163  (1.4x)|600   |1.16hrs   | A       | 0.1  |5  |∞  | 0.15/o | SGD   | bm 
CNN   |FedAVg|iid   | 99.26%,99.41%    | 19 (12.1x)| 55   (4.2x)|600   |0.41hrs   | T       | 0.1  |1  |50 | 0.2/o  | SGD   | done
CNN   |FedAVg|iid   | 99.16%,99.11%    | 21 (11.0x)| 122  (1.9x)|600   |3.86hrs   | A       | 0.1  |20 |∞  | 0.1/o  | SGD   | bm 
CNN   |FedAVg|iid   | 99.31%,99.41%    | 10 (23.0x)| 32   (7.2x)|600   |0.64hrs   | T       | 0.1  |1  |10 | 0.1/o  | SGD   | done
CNN   |FedAVg|iid   | 99.28%,99.21%    | 8  (28.8x)| 39   (5.9x)|600   |1.24hrs   | T       | 0.1  |5  |50 | 0.15/o | SGD   | done
CNN   |FedAVg|iid   | 99.28%,99.23%    | 8  (28.8x)| 37   (6.2x)|600   |4.23hrs   | T       | 0.1  |20 |50 | 0.15/o | SGD   | done
CNN   |FedAVg|iid   | 99.41%,99.38%    | 5  (46.0x)| 16  (14.4x)|600   |2.29hrs   | T       | 0.1  |5  |10 | 0.15/o | SGD   | done
CNN   |FedAVg|iid   | 99.38%,99.31%    | 5  (46.0x)| 18  (12.8x)|600   |7.48hrs   | T       | 0.1  |20 |10 | 0.1/o  | SGD   | done

##### Remarks
1. Learning curve of {E=5, B=50} is very noisy and unstable, may need a re-run

##### CNN/non-IID (selected edition)
Model |Method|Data  | Test acc (f,max) |R-98       |R-99  |T Rnd |Time      | Machine | Frac | E | B | Lr     | Optim | Status
------|------|------| --------         |-----      |----  |----  |--------  | -----   |---   |---| - | -----  | ----- | ------
CNN   |FedSGD|N-iid | 98.75%,98.83%    | 621       |      |1500  |1.12hrs   | T       | 0.1  |1  |∞  | 0.04/o | SGD   | done
CNN   |FedAVg|N-iid | 98.73%,98.81%    | 264 (2.4x)|      |600   |1.25hrs   | A       | 0.1  |5  |∞  | 0.04/o | SGD   | done
CNN   |FedAVg|N-iid | 98.54%,98.97%    | 152 (4.1x)|      |600   |0.65hrs   | A       | 0.1  |1  |50 | 0.04/o | SGD   | done
CNN   |FedAVg|N-iid | 98.79%,98.93%    | 157 (4.0x)|      |600   |3.99hrs   | A       | 0.1  |20 |∞  | 0.04/o | SGD   | done
CNN   |FedAVg|N-iid | 99.17%,99.35%    | 79  (7.9x)| 204  |600   |0.70hrs   | T       | 0.1  |1  |10 | 0.04/o | SGD   | done
CNN   |FedAVg|N-iid | 99.07%,99.15%    | 87  (7.1x)| 340  |600   |1.27hrs   | A       | 0.1  |5  |50 | 0.04/o | SGD   | done
CNN   |FedAVg|N-iid | 99.04%,99.16%    | 73  (8.5x)| 351  |600   |4.12hrs   | T       | 0.1  |20 |50 | 0.04/o | SGD   | done
CNN   |FedAVg|N-iid | 99.22%,99.30%    | 42 (14.8x)| 145  |600   |2.58hrs   | A       | 0.1  |5  |10 | 0.04/o | SGD   | done
CNN   |FedAVg|N-iid | 99.15%,99.23%    | 55 (11.3x)| 175  |600   |7.36hrs   | T       | 0.1  |20 |10 | 0.04/o | SGD   | done



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

Model |Method|Data  | T Rnd |Time      | Machine | Frac | E | B | Lr/O      |Decay  | Optim | Status
------|------|------| ----  |--------  | -----   |---   |---| - | -----     |------ | ----- | ------
CNN   |SGD   |iid   | xxxx  |hrs       | T       |      |   |100| 0.01@dp   | SGD   |       |
CNN   |FedSGD|iid   | 8000  |~8hrs     | A       | 0.1  |1  |∞  | 0.1@dp    |       | SGD   | done
CNN   |FedAVg|iid   | 4000  |9.18hrs   | T       | 0.1  |5  |50 | 0.03@dp   |       | SGD   | done



##### Varying federated setting VS SGD
* 300,000 rounds mini-batch updates used in the vanilla FL paper is too many to complete in the allowed time for now. Therefore, one may consider **100,000 mini-batch updates** instead, which is equivalent to
    1. 200 rounds (B=100) or for **100 rounds (B=50) SGD**,
    2. 200 rounds for FedAvg E=5, B=50, C=0.1
    3. 100 rounds for FedAvg E=10, B=50, C=0.1, 50 rounds E=20, B=50, C=0.1, 20 rounds for E=5, B=50, C=1.0
    4. 1000 rounds for FedAvg E=1, B=50, C=0.1
    5. 2000 rounds for FedAvg E=5, B=50, C=0.0
* The learning rates of every setting optimized to produce highest possible test acc over 100,000 mini-batch updates, while sacrificing the convergence speed as least as possible.

Model |Method|Data  | T Rnd |Time      | Machine | Frac | E | B | Lr/O         | Optim | Status
------|------|------| ----  |--------  | -----   |---   |---| - | -----        | ----- | ------
CNN   |SGD   |iid   | 100   |hrs       | T       |      |   |50 | 0.0032@dp/o  | SGD   | done
CNN   |FedAVg|iid   | 2000  |hrs       | T       | 0.0  |5  |50 | 0.02/0.32@dp | SGD   | done
CNN   |FedAVg|iid   | 1000  |hrs       | T       | 0.1  |1  |50 | 0.1@dp       | SGD   | done
CNN   |FedAVg|iid   | 200   |hrs       | T       | 0.1  |5  |50 | 0.1@dp       | SGD   | done
CNN   |FedAVg|iid   | 100   |hrs       | T       | 0.1  |10 |50 | 0.2@dp       | SGD   | done
CNN   |FedAVg|iid   | 50    |hrs       | T       | 0.1  |20 |50 | 0.2@dp       | SGD   | done
CNN   |FedAVg|iid   | 20    |hrs       | T       | 1.0  |5  |50 | 0.32@dp      | SGD   | done

##### Remarks
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


#### C. Experiment 4: Additional FedAvg with very large E and unbalanced non-IID data
Consider these additional experiment over MNIST if time permits (on 5/6-th May):
- [ ] test FedAVg E={100,200,500} B=10 C=0.1 and compare with E=1 over IID/non-IID data, using lr=0.1 (the same as E=1 B=10 C=0.1) to see the effect of very large amount of local computation.
- [ ] test FedAVg E=1 B=10 C=0.1 over unbalanced non-IID data, and compare with IID, balanced-non-IID, may use the same lr as FedAvg E=1 B=10 C=0.1 in  balanced-IID 


