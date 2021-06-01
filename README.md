# Credit Card Fraud Detection

<p align="center">
  <img src="https://github.com/alirkaya/card-fraud-detection/blob/main/Documentation/images/credit-card-lock.jpg" width="800" height="600">
</p>

<i>Can we predict fraudulent credit card transactions using data that is collected in 2 days? What kind of special treatments we have to use to analyze extremely imbalanced data sets? The project mainly answers those two questions. XGBoost Classifier achieved 97% Precision and 79% Recall scores on test data.</i>

## Table of Contents

- [Introduction](#Introduction)  
- [The Cost of Card Fraud](#The-Cost-of-Card-Fraud)  
- [About the Data](#About-the-Data)
	* [Imbalanced Data](#Imbalanced-Data)  
	* [Why Balanced Data Matter?](#Why-Balanced-Data-Matter?)
- [Exploratory Data Analysis](#Exploratory-Data-Analysis)  
- [Algorithms with Default Parameters](#Algorithms-with-Default-Parameters)  
	* [The Metrics](#The-Metrics)  
	* [Dummy Classifier](#Dummy-Classifier)  
	* [Cross-Validation Scores](#Cross-Validation-Scores)  
	* [Test Scores and Confusion Matrices](#Test-Scores-and-Confusion-Matrices)  
- [Resampling Methods](#Resampling-Methods)
- [Hyperparameter Tuning](#Hyperparameter-Tuning)  
- [Final Evaluation](#Final-Evaluation)  
- [Conclusion](#Conclusion)  

## Introduction

The project aims to predict fraudulent credit card transactions. The input matrix consists of 28 components from PCA, `time`, and `amount`. The target feature is `class`, which indicates whether a transaction is fraudulent.

In the project, I had the opportunity to exercise the following machine learning concepts: 

- Training-Validation-Test Split  
- Cross-Validation  
- Confusion Matrices  
- Evaluation Metrics  
- Imbalanced Learning  
- Binary Classification  
- Resampling Methods  
- Hyperparameter Tuning

---

[`my_module.py`](https://github.com/alirifat/fraud-detection/blob/main/Documentation/my_module.py) contains wrapping functions for `scikit-learn`, `imblearn`, `matplotlib`, `pandas` and `numpy` APIs. Functions that use `scikit-learn` API make easier to cross-validate, train and test several algorithm at once. The module contains the following functions:

* `pr_auc_score`: Calculates AUC (Area Under Curve) PR (Precision-Recall) score.
* `scoring_functions`: Returns a list of evaluation metrics such as accuracy, precision, recall, etc.
* `do_cross_validation`: Performs cross-validation manually. If `verbose=True`, provides extra information for each iteration.
* `plot_confusion_matrix`: Plots confusion matrix.
* `plot_precision_recall_curve`: Plots Precision-Recall curve.
* `plot_roc_curve`: Plots ROC (Receiver Operating Characteristics) curve.
* `calculate_statistics`: Calculates the mean and standard deviation of cross-validation scores.
* `make_df_statistics`: Puts calculated statistics into a Pandas DataFrame.
* `train_model`: Trains model(s) with the provided training data.
* `test_model`: Test model(s) with the provided test data.

---

[See the module](https://github.com/alirifat/fraud-detection/blob/main/Documentation/my_module.py)  
[See the full report](https://github.com/alirifat/fraud-detection/blob/main/Documentation/Fraud%20Detection%20Report%20PDF.pdf)  
[Find me on LinkedIn](https://www.linkedin.com/in/alirifatkaya/)  
[Go to Table of Contents](#Table-of-Contents)

---

## The Cost of Card Fraud

<img align="right" width="225" height="300" src="https://github.com/alirkaya/card-fraud-detection/blob/main/Documentation/images/nilson-fraud-report.jpg">Card fraud can be defined as making purchases via credit cards, debit cards, etc., without being authorized. It is a daily life issue that causes many people to suffer and many resources to be lost. It can happen to anyone in our society since either credit cards or debit cards mostly do our transactions.

The loss from card fraud will reach 40 Billion $ by 2027.  However, it is only the financial aspect of fraudulent transactions, which do not account for the time spent to resolve the issues and harm to society's well-being.

This project aims to detect fraudulent transactions so that it is possible to decrease the number of resources wasted and to improve the quality of life for all parties.

---

[See the full report](https://github.com/alirifat/fraud-detection/blob/main/Documentation/Fraud%20Detection%20Report%20PDF.pdf)  
[Go to Table of Contents](#Table-of-Contents)

---

## About the Data

There are 284.407 credit card transactions, 492 of whom are fraudulent, and 284.315 are genuine. In total, 1.825 transactions (27 of whom are fraudulent) have a transaction amount of 0$. In those cases, it is hard to verify that the transactions are fraudulent. For this reason, those transactions are dropped from the study.

The dataset consists of 31 features, which are `v1` - `v28`, `time`, `amount`, and `class`. The features `v1` - `v28` are the components from Principle Component Analysis. `time` feature indicates the seconds passed from the beginning of the records. `amount` feature shows the amount for each transaction. Finally, `class` feature indicates if a transaction is genuine or fraudulent.

To consolidate two-day transactions, we generated another feature, `hour`, based on the `time` feature to see how the transactions were distributed hourly.

---

[Download the data set](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
[Go to Table of Contents](#Table-of-Contents)

---

## Imbalanced Data

Imbalanced data refers to the case where classes are not represented equally in a data set. The _ideal_ balanced data set should have an equal representation of each class. If it is violated, then we have an imbalanced data set.

It is possible to confuse the concepts of imbalanced classification and unbalanced classification. While the former refers to "_a classification predicting modeling problem where the number of examples across the classes is not equal_," and the latter refers to the deterioration of the data set's balanced structure. For this reason, using two concepts interchangeably will lead to misunderstanding of the research problem [(Brownlee, 2020)](https://machinelearningmastery.com/what-is-imbalanced-classification/).

In the data set, the ratio of fraudulent transactions to genuine transactions is 1:578. In other words, there are 578 genuine transactions for each fraudulent transaction. However, it is the industry standard. The fraudulent transactions are expected to be a tiny proportion of all credit card transactions.

---

[Go to Table of Contents](#Table-of-Contents)

---

## Why Balanced Data Matter?

As discussed [here](https://stats.stackexchange.com/questions/227088/when-should-i-balance-classes-in-a-training-data-set), the problem is not the difference in the number of observations for each class. However, machine learning algorithms will not recognize a positive class pattern if there isn't enough observation. For this reason, having an imbalanced dataset may not cause any problem; unless there is enough information retained for the positive class. For a somewhat technical explanation of imbalanced class distribution, an interested reader can refer to [Rocca (2019)](https://towardsdatascience.com/handling-imbalanced-datasets-in-machine-learning-7a0e84220f28).

According to [Brownlee (2020)](https://machinelearningmastery.com/what-is-imbalanced-classification/), '_most machine learning algorithms for classification predictive models are designed and demonstrated on problems that assume an equal distribution of classes_.' Thus, having an imbalanced data set will let the minority class disguise itself among the majority class and not be detected effectively. For this reason, imbalanced data sets may require special treatments before predictive analysis.

---

[Go to Table of Contents](#Table-of-Contents)

---

## Exploratory Data Analysis

The heatmap shows the correlation among features by color-coded cells. The features `v1`-`v28` are acquired through PCA so that they are orthogonal to each other. In other words, there is no correlation among those features.

<p align="center">
  <img src="https://github.com/alirkaya/card-fraud-detection/blob/main/Documentation/images/correlation_heatmap.jpg" width="800" height="600">
</p>


The correlation coefficients for `v7`-`amount` and `v20`-`amount` are the highest among all others. For this reason, it may be a good idea to have a closer look at those relationships. However, in the scatterplots, we see that the coefficients are inflated due to outliers. 

<p align="center">
  <img src="https://github.com/alirkaya/card-fraud-detection/blob/main/Documentation/images/scatterplots.jpg" width="800" height="300">
</p>


Because univariate analysis didn't provide any useful results. We will move on to the bivariate analysis, in which we will look at the relationships between each input feature and the target feature. Further details can be seen in the [notebook](https://github.com/alirifat/fraud-detection/blob/master/01_exploratory_data_analysis.ipynb).

<p align="center">
  <img src="https://github.com/alirkaya/card-fraud-detection/blob/main/Documentation/images/bivariate/V17.jpg" width="800" height="300">
</p>


Finally, we will compare the histograms for fraudulent transactions and genuine transactions. To consolidate transactions into one day, we created a new feature, `hour`, which indicates when the transaction took place. 

<p align="center">
  <img src="https://github.com/alirkaya/card-fraud-detection/blob/main/Documentation/images/hour_histogram.jpg" width="800" height="600">
</p>


---

[See the notebook](https://github.com/alirifat/fraud-detection/blob/main/01_exploratory_data_analysis.ipynb)  
[Go to Table of Contents](#Table-of-Contents)

---

## Algorithms with Default Parameters

We used the following supervised machine learning algorithms: __Logistic Regression__, __k-Nearest Neighbors__, __Extra Trees__, and __XGBoost__.

We separated 30% of the data as hold-out test set and performed stratified 5-fold cross-validation on the training set. 

#### The Metrics

Initially, we used the following evaluation metrics: **accuracy**, **precision**, **recall**, **specificity**, **f1-score**, **f2-score**, **geometric mean**, **matthews correlation coefficient (MCC)**, **precision-recall (PR) curve**, and **ROC curve**.

However, the evaluation metrics may not provide accurate results due to the data set's extremely imbalanced structure. For this reason, we spot-checked the metrics using Dummy Classifier.

#### Cross-Validation Scores

Cross-Validation lets us reduce the impact of *luck* in our studies by training and testing the models from scratch with different data parts and getting an interval for future results.

<p align="center">
  <img src="https://github.com/alirkaya/card-fraud-detection/blob/main/Documentation/images/cv_scores.jpg" width="800" height="600">
</p>


While algorithms have closer **MCC** and AUC (Area Under Curve) **PR** scores, they have almost perfect AUC **ROC** scores. [Brownlee (2020)](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-imbalanced-classification/) states that **ROC** may report optimistic results under the condition of extreme class imbalance. In this study, we believe that AUC **ROC** scores reflect that optimism. For this reason, we will drop **ROC** from the study.

#### Resampling Methods

The results are calculated by taking the average of all algorithms' scores for each metric. The reasoning is that an increase in the data quality will be visible on the metrics so that __SMOTEENN__ will generate higher scores on average. However, we couldn't see such improvement. For this reason, we will use the original data set.

<p align="center">
  <img src="https://github.com/alirkaya/card-fraud-detection/blob/main/Documentation/images/cv_scores_resampling.jpg" width="600" height="400">
</p>



---

[See the notebook](https://github.com/alirifat/fraud-detection/blob/main/02_algorithms_with_default_parameters.ipynb)  
[Go to Table of Contents](#Table-of-Contents)

---

## Hyperparameter Tuning

We used GridSearchCv. which goes over every possible combination of parameters in the parameter space to find the best performing set of parameters. 

|                                 | Hyperparameter Tuning | Hyperparameter Tuning with SMOTEENN | **Change** |
| ------------------------------- | :-------------------: | :---------------------------------: | :--------: |
| Logistic Regression             |       0.745125        |              0.735264               | -0.009861  |
| Extra Trees Classifier          |       0.863069        |              0.839344               | -0.023725  |
| XGBoost Classifier              |       0.854038        |              0.821997               | -0.032041  |
| k-Nearest Neighbors  Classifier |       0.872836        |              0.681403               | -0.191433  |

---

[Go to Table of Contents](#Table-of-Contents)

---

## Conclusion

* Achieved **95% Precision** and **73%** recall scores.
* A narrow set of hyperparameters were tested and the best performing pair of parameters are selected to be tested for final evaluation.
* Resampling methods didn't favor the algorithms.
* Voting classifier is as good as Extra Trees classifier.
* Even though algorithms have slight differences the best performing one is Extra Trees, which provided the best results in Confusion Matrix.



<p align="center">
  <img src="https://github.com/alirkaya/card-fraud-detection/blob/main/Documentation/images/confusion_matrices.PNG" width="600" height="600">
</p>



---

[Go to Table of Contents](#Table-of-Contents)

---
