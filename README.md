# Credit Card Fraud Detection

<p align="center">
  <img src="https://github.com/alirifat/fraud-detection/blob/main/Documentation/figures/credit-card-lock.jpg" width="800" height="600">
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

<img align="right" width="225" height="300" src="https://github.com/alirifat/fraud-detection/blob/main/Documentation/figures/nilson-fraud-report.jpg">Card fraud can be defined as making purchases via credit cards, debit cards, etc., without being authorized. It is a daily life issue that causes many people to suffer and many resources to be lost. It can happen to anyone in our society since either credit cards or debit cards mostly do our transactions.

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
  <img src="https://github.com/alirifat/fraud-detection/blob/main/Documentation/figures/correlation-matrix.JPG" width="800" height="600">
</p>

The correlation coefficients for `v7`-`amount` and `v20`-`amount` are the highest among all others. For this reason, it may be a good idea to have a closer look at those relationships. However, in the scatterplots, we see that the coefficients are inflated due to outliers. 

<p align="center">
  <img src="https://github.com/alirifat/fraud-detection/blob/main/Documentation/figures/scatterplots.JPG" width="800" height="300">
</p>

Because univariate analysis didn't provide any useful results. We will move on to the bivariate analysis, in which we will look at the relationships between each input feature and the target feature. Further details can be seen in the [notebook](https://github.com/alirifat/fraud-detection/blob/master/01_exploratory_data_analysis.ipynb).

<p align="center">
  <img src="https://github.com/alirifat/fraud-detection/blob/main/Documentation/figures/bivariate-analysis.JPG" width="800" height="300">
</p>

Finally, we will compare the histograms for fraudulent transactions and genuine transactions. To consolidate transactions into one day, we created a new feature, `hour`, which indicates when the transaction took place. 

<p align="center">
  <img src="https://github.com/alirifat/fraud-detection/blob/main/Documentation/figures/histograms.JPG" width="800" height="600">
</p>

---

[See the notebook](https://github.com/alirifat/fraud-detection/blob/main/01_exploratory_data_analysis.ipynb)  
[Go to Table of Contents](#Table-of-Contents)

---

## Algorithms with Default Parameters

We used the following supervised machine learning algorithms: __Logistic Regression__, __Naïve Bayes__, __k-Nearest Neighbors__, __Decision Tree__, __Random Forest__, __Extra Trees__, __AdaBoost__, __Gradient Boosting__, and __XGBoost__.

We separated 30% of the data as hold-out test set and performed stratified 5-fold cross-validation on the training set. 

#### The Metrics

Initially, we used the following evaluation metrics: **accuracy**, **precision**, **recall**, **specificity**, **f1-score**, **f2-score**, **geometric mean**, **matthews correlation coefficient (MCC)**, **precision-recall (PR) curve**, and **ROC curve**.

However, the evaluation metrics may not provide accurate results due to the data set's extremely imbalanced structure. For this reason, we spot-checked the metrics using Dummy Classifier.

#### Dummy Classifier

Dummy Classifier doesn't consider the input matrix while making predictions. For this reason, it can provide the necessary insight into the evaluation metrics. In this step, we compared Dummy Classifier with Logistic Regression.

<p align="center">
  <img src="https://github.com/alirifat/fraud-detection/blob/main/Documentation/figures/dummy-logistic.JPG" width="600" height="500">
</p>

We see that **accuracy** and **specificity** are not useful metrics for extremely imbalanced data sets from the figure. Moreover, there is a considerable gap between **precision** and **recall**, which will affect the derivative metrics' performance, such as **f1-score**, **f2-score**, and **geometric mean**. For this reason, we will continue with __MCC__, __PR curve__, and **ROC curve**.

#### Cross-Validation Scores

Cross-Validation lets us reduce the impact of *luck* in our studies by training and testing the models from scratch with different data parts and getting an interval for future results.

<p align="center">
  <img src="https://github.com/alirifat/fraud-detection/blob/main/Documentation/figures/validation-scores.JPG" width="800" height="600">
</p>

While algorithms have closer **MCC** and AUC (Area Under Curve) **PR** scores, they have almost perfect AUC **ROC** scores. **Naïve Bayes** has one of the highest AUC **ROC** scores, but it is not consistent with other metrics. [Brownlee (2020)](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-imbalanced-classification/) states that **ROC** may report optimistic results under the condition of extreme class imbalance. In this study, we believe that AUC **ROC** scores reflect that optimism. For this reason, we will drop **ROC** from the study.

Finally, **Gradient Boosting** has the highest error, which means its scores vary in a large interval, but it is not the desired outcome.

#### Resampling Methods

The results are calculated by taking the average of all algorithms' scores for each metric. The reasoning is that an increase in the data quality will be visible on the metrics so that __SMOTEENN__ will generate higher scores on average. However, we couldn't see such improvement. For this reason, we will use the original data set.

<p align="center">
  <img src="https://github.com/alirifat/fraud-detection/blob/main/Documentation/figures/resampling.JPG" width="600" height="400">
</p>

---

[See the notebook](https://github.com/alirifat/fraud-detection/blob/main/02_algorithms_with_default_parameters.ipynb)  
[Go to Table of Contents](#Table-of-Contents)

---

## Hyperparameter Tuning

We used GridSearchCv. which goes over every possible combination of parameters in the parameter space to find the best performing set of parameters. **Gradient Boosting** has the most significant benefit from hyperparameter tuning, which puts it into the race again.

|                                 | **Default  Parameters** | **Tuned Parameters** | **Change** |
| ------------------------------- | :---------------------: | :------------------: | :--------: |
| Logistic Regression             |          0.756          |        0.758         |   0.002    |
| Decision Tree Classifier        |          0.734          |        0.761         |   0.027    |
| Random Forest Classifier        |          0.854          |        0.860         |   0.060    |
| Extra Trees Classifier          |          0.861          |        0.864         |   0.003    |
| AdaBoost Classifier             |          0.761          |        0.829         |   0.068    |
| Gradient Boosting Classifier    |          0.604          |        0.828         | __0.224__  |
| XGBoost Classifier              |          0.850          |        0.861         |   0.011    |
| k-Nearest Neighbors  Classifier |          0.852          |        0.859         |   0.007    |

---

[See the notebook1](https://github.com/alirifat/fraud-detection/blob/main/03_hyperparameter_optimization.ipynb)  
[See the notebook2](https://github.com/alirifat/fraud-detection/blob/main/03_knn_hyperparameter_optimization.ipynb)  
[Go to Table of Contents](#Table-of-Contents)

---

## Final Evaluation

We used a narrow set of hyperparameters and selected the best performing combinations for the final evaluation. We did the final evaluation using the hold-out test set. For this reason, these scores reflect the algorithms' performances on brand-new data. 

<p align="center">
  <img src="https://github.com/alirifat/fraud-detection/blob/main/Documentation/figures/test-scores.JPG" width="800" height="600">
</p>

We did our evaluation based on the evaluation metrics and the confusion matrix. We find that __XGBoost__ algorithm with tuned parameter is the best performing algorithm. We achieved 97% Precision and 79% Recall scores on test data.

<p align="center">
  <img src="https://github.com/alirifat/fraud-detection/blob/main/Documentation/figures/confusion-matrix.JPG" width="400" height="300">
</p>

---

[See the notebook](https://github.com/alirifat/fraud-detection/blob/main/04_final_evaluation.ipynb)  
[Go to Table of Contents](#Table-of-Contents)

---

## Conclusion

In this project, we aimed to predict fraudulent transactions. The [data](#About-the-Data) set consists of 28 components acquired from PCA, `time`, `amount`, and `class` features. It also has a ratio of 1:578 for fraudulent transactions.

In the preprocessing stage, we dropped the 0$ transactions. Later, we separated data into the training (70%), and test (30%) sets.

Initially, we selected [ten evaluation metrics](#The-Metrics) and spot-checked them by comparing Dummy Classifier and Logistic Regression. As a result, we decided to continue with __matthews correlation coefficient (MCC)__, __Precision-Recall (PR) curve__, and __ROC curve__.

On the other hand, we had [a broad set of algorithms](#Algorithms-with-Default-Parameters). We used stratified 5-fold [cross-validation](#Cross-Validation-Scores) on the training data. We see that __ROC__ behaves optimistically under extreme class distribution. For this reason, it was dropped from the study. 

In the next step, we [resampled](#Resampling-Methods) the data using SMOTEENN to improve the class distribution. However, we couldn't see any improvement. Thus, we decided to go with the original data.

Additionally, we [used GridSearchCV](#Hyperparameter-Tuning) to find the best combination of parameters. In the [final step](#Final-Evaluation), we trained the algorithms using the training data (70%) and test them on the test data (30%). Because, we used hold-out test sample, the scores reflect the algorithms' performances on brand-new data.

Finally, we achieved achieved  88% MCC and 81% PR scores (97% Precision and 79% Recall) using XGBoost algorithm.

---

[Go to Table of Contents](#Table-of-Contents)

---
