"""Prepared by [Ali Rifat Kaya](https://www.linkedin.com/in/alirifatkaya/)
"""


def pr_auc_score(y_test, predicted_probabilities):
    """Return AUCPR (Area Under Curve Precision-Recall) score

    Parameters
    ----------

    y_test : Test set target values
        Example:
        # df is a pandas dataframe with features and target variable
        # where 'Class' is the target variable
        >>> import pandas as pd
        >>> from sklearn.model_selection import train_test_split

        >>> X = df.drop('Class', axis=1).values # input matrix
        >>> y = df['Class'].values # target array
        >>> X_train, X_test, y_train, y_test = train_test_split(X,
                                                                y,
                                                                test_size=0.3,
                                                                random_state=1)
        >>> # y_test is the target values for test set


    predicted_probabilities : Predicted probabilities for positive class
        Example:
        >>> from sklearn.linear_model import LogisticRegression
        >>> lr = LogisticRegression()
        >>> lr.fit(X_train, y_train)
        >>> predicted_probabilities = lr.predict_proba(X_test)[:, 1]


    Returns
    -------

    auc_score : The AUCPR score for the given target values and probabilities
    """
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import auc

    precision, recall, threshold = precision_recall_curve(y_test,
                                                          predicted_probabilities)
    auc_score = auc(recall, precision)

    return auc_score


def scoring_functions():
    """Returns a list of scoring functions as a list
        * Accuracy Score
        * Precision Score
        * Recall Score
        * Specificity Score
        * F1 Score
        * F2 Score
        * Matthews Correlation Coefficient
        * Geometric Mean Score
        * AUCPR Score
        * AUCROC Score

    Returns
    -------

    List of scoring fucntions
        Example:
        >>> list_of_scoring_functions = scores()
        >>> for scoring_function in list_of_scoring_functions:
        ...     print(scoring_function)
        ### prints
        # accuracy_score
        # precision_score
        # recall_score
        # specificity_score
        # f1_score
        # fbeta_score
        # geometric_mean_score
        # matthews_corrcoef
        # roc_auc_score
        # pr_auc_score
    """
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from imblearn.metrics import specificity_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import fbeta_score
    from imblearn.metrics import geometric_mean_score
    from sklearn.metrics import matthews_corrcoef
    from sklearn.metrics import roc_auc_score

    list_of_scoring_functions = [
        accuracy_score,
        precision_score,
        recall_score,
        specificity_score,
        f1_score,
        fbeta_score,
        geometric_mean_score,
        matthews_corrcoef,
        roc_auc_score,
        pr_auc_score
    ]

    return list_of_scoring_functions


def do_cross_validation(X, y, estimators, cv=None, resample=None, scalers=[False], verbose=True, sleep_time=None):
    """ Return Cross-Validation score for each fold by fitting the model from
    scratch.

    Parameters
    ----------

    X: The input matrix
        Example:
        # df is a pandas dataframe with features and target variable
        # where 'Class' is the target variable
        >>> X = df.drop('Class', axis=1).values


    y: The target array
        Example:
        # df is a pandas dataframe with features and target variable
        # where the 'Class' is the target variable
        >>> y = df['Class'].values


    estimators: A list of tuple(s) where the tuple is ('estimator_name', estimator)
        Example:
        >>> from sklearn.linear_model import LogisticRegresion
        >>> lr = LogisticRegresion()
        >>> estimators = [('Logistic Regression', lr)]

        >>> from sklearn.linear_model import LogisticRegresion
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> lr = LogisticRegresion()
        >>> rf = RandomForestClassifier()
        >>> estimators = [('Logistic Regression', lr),
        ...               ('Random Forest Classifier', rf)]


    cv: Cross-Validation object. If no cross-validation object is passed to `cv`,
        then cv is `StratifiedKFold(n_splits=5, shuffle=True, random_state=1)`
        by default.
        Example:
        >>> from sklearn.model_selection import StratifiedKFold
        >>> cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)


    resample: if True, resample the training data and fits the models using the
              resampled training data. Do NOT touch to validation data.
              Default value is `None`.
        Example:
        >>> from imblearn.over_sampling import SMOTE
        >>> from sklearn.linear_model import LogisticRegresion
        >>> smote = SMOTE()
        >>> resample = [('SMOTE', smote)]
        >>> lr = LogisticRegresion()
        >>> estimators = [('Logistic Regression', lr)]
        >>> do_cross_validation(X, y, estimators=estimators, cv, resample=resample, scaler=[True], verbose=False)


    scalers: An array of boolean values, each value is for the corresponding
             estimator.
             Default value is `[False]`
        Example:
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> lr = LogisticRegression()
        >>> rf = RandomForestClassifier()
        >>> models = [lr, rf]
        >>> scalers = [True, False]
        >>> cv_results = do_cross_validation(X, y, estimators=models,
                                             cv, scalers=scalers, verbose=False)


    print: if True, prints out information about each fold such as size of the
           training data and test data, AUCPR and AUCROC scores for each fold,
           and predicted labels.
           Default value is `True`.


    sleep_time: Sleeping time in seconds between each iteration
        Example:
        >>> sleep_time=1800 # 30 mins
        >>> cv_results = do_cross_validation(X, y, estimators=models,
                                             cv, scalers=scalers, verbose=False,
                                             sleep_time=sleep_time)



    Returns
    -------

    Nested dictionary of results with
        * precisions and recalls to plot precision-recall curve
        * fpr and tpr to plot roc curve
        Example:
        >>> {
        'Logistic Regression' : {
            'accuracy_score' : [], # cross validation accuracy scores as a list
            ...
            'tprs' : [] # cross validation tpr for each fold
            }
        }


    Verbose
    ------

    For each fold of cross-validation, prints the followings:
        * The estimator
        * Training set and validation set sizes
        * AUCPR score for training and validation sets
        * AUCROC score for training and validation sets
        * Number of True Positives in the validation set
        * Number of False Positives in the validation set
    """
    from sklearn.preprocessing import scale
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import fbeta_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import roc_curve
    from sklearn.base import clone
    from time import sleep

    scores = {}
    list_of_scoring_functions = scoring_functions()
    metrics = ['accuracy_score', 'precision_score', 'recall_score',
               'specificity_score', 'f1_score', 'f2_score',
               'geometric_mean_score', 'matthews_corrcoef', 'roc_auc_score',
               'pr_auc_score']

    if not cv:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    i = 0  # tracks the folds
    for train_idx, validation_idx in cv.split(X, y):
        X_train, X_validation = X[train_idx], X[validation_idx]
        y_train, y_validation = y[train_idx], y[validation_idx]

        X_train_copy = X_train.copy()
        y_train_copy = y_train.copy()
        X_validation_copy = X_validation.copy()

        if resample:
            if verbose:
                print('Fold {}:'.format(i + 1))

            for name, method in resample:
                X_train_copy_resample, y_train_copy_resample = method.fit_resample(
                    X_train_copy, y_train_copy)
                if verbose:
                    print('\n'+name)
                    print('-' * 81)
                    print('Number of transactions in the original training dataset:', X_train_copy.shape[0])
                    print('Number of transactions in the resampled training dataset:',
                          X_train_copy_resample.shape[0])
                    print('-' * 81)
                    print('Number of Fraudulent Transactions in the original training dataset:',
                          y_train_copy.sum())
                    print('Number of Fraudulent Transactions in the resampled training dataset',
                          y_train_copy_resample.sum())
                    print('=' * 81)

                for estimator, scaler in zip(estimators, scalers):
                    ml_name, ml = estimator
                    estimator_ = clone(ml)
                    if scaler:
                        X_train_copy_resample_scaled = scale(
                            X_train_copy_resample)
                        X_validation_scaled = scale(X_validation_copy)
                        estimator_.fit(
                            X_train_copy_resample_scaled, y_train_copy_resample)
                        preds = estimator_.predict(X_validation_scaled)
                        probas_training = estimator_.predict_proba(X_train_copy_resample_scaled)[
                            :, 1]
                        probas = estimator_.predict_proba(
                            X_validation_scaled)[:, 1]
                    else:
                        estimator_.fit(X_train_copy_resample,
                                       y_train_copy_resample)
                        preds = estimator_.predict(X_validation_copy)
                        probas_training = estimator_.predict_proba(
                            X_train_copy_resample)[:, 1]
                        probas = estimator_.predict_proba(
                            X_validation_copy)[:, 1]

                    precision, recall, threshold = precision_recall_curve(
                        y_validation, probas)
                    fpr, tpr, threshold = roc_curve(y_validation, probas)
                    tn, fp, fn, tp = confusion_matrix(y_validation, preds).ravel()

                    if verbose:
                        print('\n' + ml_name + ' with ' + name)
                        print('-' * 81)
                        print('Training data AUCPR score: {}'.format(
                            pr_auc_score(y_train_copy_resample, probas_training)))
                        print('Validation data AUCPR score: {}'.format(
                            pr_auc_score(y_validation, probas)))

                        print('\nTraining data AUCROC score: {}'.format(
                            roc_auc_score(y_train_copy_resample, probas_training)))
                        print('Validation data AUCROC score: {}'.format(
                            roc_auc_score(y_validation, probas)))
                        print('-' * 81)
                        print('There are {} fraudulent transactions in the validation '
                              'set'.format(y_validation.sum()))
                        print('{} out of {} predicted fraudulent transactions '
                              'are true fraudulent transactions'.format(
                                  tp, fp + tp))
                        print()

                    key_ = ml_name + '_' + name
                    if key_ not in scores.keys():
                        scores[key_] = {}

                    plots = ['precisions', 'recalls', 'fprs', 'tprs']
                    for key in plots:
                        if key not in scores[key_]:
                            scores[key_][key] = []

                    scores[key_]['precisions'].append(precision)
                    scores[key_]['recalls'].append(recall)
                    scores[key_]['fprs'].append(fpr)
                    scores[key_]['tprs'].append(tpr)

                    for metric_name, metric in zip(metrics, list_of_scoring_functions):
                        if metric_name not in scores[key_].keys():
                            scores[key_][metric_name] = []
                        if metric in [roc_auc_score, pr_auc_score]:
                            scores[key_][metric_name].append(metric(y_validation,
                                                                       probas))
                        elif metric == fbeta_score:
                            scores[key_][metric_name].append(metric(y_validation,
                                                                       preds,
                                                                       beta=2))
                        else:
                            scores[key_][metric_name].append(metric(y_validation,
                                                                       preds))
                    if sleep_time:
                        print('sleeping... {} seconds'.format(sleep_time))
                        sleep(sleep_time)

            if verbose:
                print()
        else:
            if verbose:
                print('Fold {}:'.format(i + 1))
                print('\nNumber of Observations in the Training Data: {}'
                      .format(X_train.shape[0]))
                print('Number of Observations in the Validation Data: {}:'
                      .format(y_validation.shape[0]))
                print('=' * 81)
            for estimator, scaler in zip(estimators, scalers):
                ml_name, ml = estimator
                estimator_ = clone(ml)
                if scaler:
                    X_train_scaled = scale(X_train_copy)
                    X_validation_scaled = scale(X_validation_copy)
                    estimator_.fit(X_train_scaled, y_train)
                    preds = estimator_.predict(X_validation_scaled)
                    probas_training = estimator_.predict_proba(X_train_scaled)[
                        :, 1]
                    probas = estimator_.predict_proba(X_validation_scaled)[:, 1]
                else:
                    estimator_.fit(X_train, y_train)
                    preds = estimator_.predict(X_validation)
                    probas_training = estimator_.predict_proba(X_train)[:, 1]
                    probas = estimator_.predict_proba(X_validation)[:, 1]

                precision, recall, threshold = precision_recall_curve(
                    y_validation, probas)
                fpr, tpr, threshold = roc_curve(y_validation, probas)
                tn, fp, fn, tp = confusion_matrix(y_validation, preds).ravel()

                if verbose:
                    print('\n' + ml_name)
                    print(('-' * 81))
                    print(('Training data AUCPR score: {}'.format(
                        pr_auc_score(y_train, probas_training))))
                    print(('Validation data AUCPR score: {}'.format(
                        pr_auc_score(y_validation, probas))))

                    print(('\nTraining data AUCROC score: {}'.format(
                        roc_auc_score(y_train, probas_training))))
                    print(('Validation data AUCROC score: {}'.format(
                        roc_auc_score(y_validation, probas))))
                    print(('-' * 81))
                    print('There are {} fraudulent transactions in the validation '
                          'set'.format(y_validation.sum()))
                    print('{} out of {} predicted fraudulent transactions '
                          'are true fraudulent transactions'.format(
                              tp, fp + tp))
                    print()

                if ml_name not in scores.keys():
                    scores[ml_name] = {}

                plots = ['precisions', 'recalls', 'fprs', 'tprs']
                for key in plots:
                    if key not in scores[ml_name]:
                        scores[ml_name][key] = []

                scores[ml_name]['precisions'].append(precision)
                scores[ml_name]['recalls'].append(recall)
                scores[ml_name]['fprs'].append(fpr)
                scores[ml_name]['tprs'].append(tpr)

                for metric_name, metric in zip(metrics, list_of_scoring_functions):
                    if metric_name not in scores[ml_name].keys():
                        scores[ml_name][metric_name] = []
                    if metric in [roc_auc_score, pr_auc_score]:
                        scores[ml_name][metric_name].append(metric(y_validation,
                                                                   probas))
                    elif metric == fbeta_score:
                        scores[ml_name][metric_name].append(metric(y_validation,
                                                                   preds,
                                                                   beta=2))
                    else:
                        scores[ml_name][metric_name].append(metric(y_validation,
                                                                   preds))
                if sleep_time:
                    print('sleeping... {} seconds'.format(sleep_time))
                    sleep(sleep_time)
        if verbose:
            print()
        i += 1
    if verbose:
        print('=' * 81)
        print('=' * 81)

    return scores


def plot_confusion_matrix(y, predictions, title=None, ax=None, cmap='Purples', cbar=False):
    """Plots Confusion Matrix

    Parameters
    ----------

    y: The target array
        Example:
        # df is a pandas dataframe with features and target variable
        # where the 'Class' is the target variable
        >>> y = df['Class'].values

    predictions: The predicted labels
        Example:
        # df is a pandas dataframe with features and target variable
        # where the 'Class' is the target variable
        >>> from sklearn.linear_model import LogisticRegresion
        >>> X = df.drop('Class', axis=1).values
        >>> y = df['Class'].values
        >>> lr = LogisticRegresion()
        >>> lr.fit(X_train, y_train)
        >>> predictions = lr.predict(X_test) # predicted labels

    title: Title of the plot
        Example:
        >>> title = 'Logistic Regression'
        >>> plot_confusion_matrix(y, predictions, title=title, ax, cmap, cbar)

    ax: An axis object
        Example:
        >>> fig, ax = plt.subplots()
        >>> plot_confusion_matrix(y, predictions, title, ax=ax)

    cmap: The color map for the confusion matrix
        Example:
        >>> import matplotlib.pyplot as plt
        >>> plt.colormaps() # prints all available color maps

    cbar: If True shows the color bar next to the confusion matrix
        Example
        >>> plot_confusion_matrix(y, predictions, title, ax=ax, cbar=True)

    Returns
    -------

    ax: Axes object
    """

    from pandas import DataFrame
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    from seaborn import heatmap

    if ax is None:
        ax = plt.gca()

    cm_df = DataFrame(confusion_matrix(y, predictions))

    # Use a seaborn heatmap to plot confusion matrices
    # The dataframe is transposed to make Actual values on x-axis and
    # predicted values on y-axis
    # annot = True includes the numbers in each box
    # vmin and vmax just adjusts the color value
    heatmap(cm_df.T,
            annot=True,
            annot_kws={"size": 15},
            cmap=cmap,
            vmin=0,
            vmax=800,
            fmt='.0f',
            linewidths=1,
            linecolor="white",
            cbar=cbar,
            xticklabels=["Genuine", "Fraud"],
            yticklabels=["Genuine", "Fraud"],
            ax=ax)

    # adjusts the heights of the top and bottom squares of the heatmap
    # matplotlib 3.1.1 has a bug that shows only the half of the top
    # and bottom rows of the heatmap
    # bottom, top = ax.get_ylim()
    # _ = ax.set_ylim(bottom + 0.5, top - 0.5)

    # ax.set_ylabel("Predicted", fontweight='bold', fontsize=15)
    # ax.set_xlabel("Actual", fontweight='bold', fontsize=15)
    ax.set_xticklabels(["Genuine", "Fraud"], fontsize=13)
    ax.set_yticklabels(["Genuine", "Fraud"], fontsize=13)
    ax.set_title(title, fontweight='bold', pad=5)

    return ax


def plot_precision_recall_curve(y_test, precisions, recalls, title, ax=None):
    """Plots Precision-Recall Curve

    Parameters
    ----------

    y: The target array of the test set
        Example:
        # df is a pandas dataframe with features and target variable
        # where the 'Class' is the target variable
        >>> X = df.drop('Class', axis=1).values
        >>> y = df.Class.values
        >>> X_train, X_test, y_train, y_test = train_test_split(X,
                                                                y,
                                                                test_size=0.3,
                                                                random_state=1)


    precisions: Precision score for each threshold
        Example:
        # df is a pandas dataframe with features and target variable
        # where 'Class' is the target variable
        >>> from sklearn.linear_model import LogisticRegresion
        >>> from sklearn.metrics import precision_recall_curve
        >>> X = df.drop('Class', axis=1).values
        >>> y = df.Class.values
        >>> X_train, X_test, y_train, y_test = train_test_split(X,
                                                                y,
                                                                test_size=0.3,
                                                                random_state=1)
        >>> lr = LogisticRegresion()
        >>> lr.fit(X_train, y_train)
        >>> predicted_probabilities = lr.predict_proba(X_test)[:, 1]
        >>> precision, _, _ = precision_recall_curve(y, predicted_probabilities)


    recalls: Recall score for each threshold
        Example:
        # df is a pandas dataframe with features and target variable
        # where 'Class' is the target variable
        >>> from sklearn.linear_model import LogisticRegresion
        >>> from sklearn.metrics import precision_recall_curve
        >>> X = df.drop('Class', axis=1).values
        >>> y = df.Class.values
        >>> X_train, X_test, y_train, y_test = train_test_split(X,
                                                                y,
                                                                test_size=0.3,
                                                                random_state=1)
        >>> lr = LogisticRegresion()
        >>> lr.fit(X_train, y_train)
        >>> predicted_probabilities = lr.predict_proba(X_test)[:, 1]
        >>> _, recall, _ = precision_recall_curve(y, predicted_probabilities)


    title: Title of the plot
        Example:
        >>> title = 'Logistic Regression'
        >>> plot_precision_recall_curve(precisions, recalls, title=title, ax)


    ax: An axis object
        Example:
        >>> fig, ax = plt.subplots()
        >>> plot_confusion_matrix(y, predictions, title, ax=ax)


    Returns
    -------

    ax: Axes object
    """

    from numpy import linspace
    from numpy import interp
    from numpy import mean
    from numpy import std
    from numpy import minimum
    from numpy import maximum
    from sklearn.metrics import auc
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    # Metrics
    prs = []
    aucs = []
    mean_recall = linspace(0, 1, 100)

    # plots PR curve for each fold
    i = 0
    for precision, recall in zip(precisions, recalls):
        prs.append(interp(mean_recall, precision, recall))
        pr_auc = auc(recall, precision)
        aucs.append(pr_auc)
        ax.plot(recall,
                precision,
                lw=3,
                alpha=0.5,
                label='Fold %d (AUCPR = %0.2f)' % (i + 1, pr_auc))
        i += 1

    # plots the mean AUCPR curve
    ax.axhline(y_test.sum() / y_test.shape[0],
               linestyle='--',
               alpha=0.8,
               label='No Skill')
    mean_precision = mean(prs, axis=0)
    mean_auc = auc(mean_recall, mean_precision)
    std_auc = std(aucs)
    ax.plot(mean_precision,
            mean_recall,
            color='navy',
            label=r'Mean (AUCPR = %0.3f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=4)

    ax.set_title(title)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.legend(fontsize='xx-small')

    return ax


def plot_roc_curve(fprs, tprs, title, ax=None):
    """Plots ROC (Receiver Operating Curve)

    Parameters
    ----------

    fprs: False Positive Rate for each threshold
        Example:
        # df is a pandas dataframe with features and target variable
        # where 'Class' is the target variable
        >>> from sklearn.linear_model import LogisticRegresion
        >>> from sklearn.metrics import precision_recall_curve
        >>> X = df.drop('Class', axis=1).values
        >>> y = df.Class.values
        >>> X_train, X_test, y_train, y_test = train_test_split(X,
                                                                y,
                                                                test_size=0.3,
                                                                random_state=1)
        >>> lr = LogisticRegresion()
        >>> lr.fit(X_train, y_train)
        >>> predicted_probabilities = lr.predict_proba(X_test)[:, 1]
        >>> fpr, _, _ = roc_curve(y, predicted_probabilities)


    tprs: True Positive Rate for each threshold
        Example:
        # df is a pandas dataframe with features and target variable
        # where 'Class' is the target variable
        >>> from sklearn.linear_model import LogisticRegresion
        >>> from sklearn.metrics import precision_recall_curve
        >>> X = df.drop('Class', axis=1).values
        >>> y = df.Class.values
        >>> X_train, X_test, y_train, y_test = train_test_split(X,
                                                                y,
                                                                test_size=0.3,
                                                                random_state=1)
        >>> lr = LogisticRegresion()
        >>> lr.fit(X_train, y_train)
        >>> predicted_probabilities = lr.predict_proba(X_test)[:, 1]
        >>> _, tpr, _ = roc_curve(y, predicted_probabilities)


    title: Title of the plot
        Example:
        >>> title = 'Logistic Regression'
        >>> plot_precision_recall_curve(precisions, recalls, title=title, ax)


    ax: An axis object
        Example:
        >>> fig, ax = plt.subplots()
        >>> plot_confusion_matrix(y, predictions, title, ax=ax)


    Returns
    -------

    ax: Axes object
    """

    from numpy import linspace
    from numpy import interp
    from numpy import mean
    from numpy import std
    from numpy import minimum
    from numpy import maximum
    from sklearn.metrics import auc
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    # Metrics
    tprs_ = []
    aucs = []
    mean_fpr = linspace(0, 1, 100)

    # plots ROC curves for each fold
    i = 0
    for fpr, tpr in zip(fprs, tprs):
        interp_tpr = interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs_.append(interp_tpr)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr,
                tpr,
                lw=3,
                alpha=0.5,
                label='ROC Fold %d (AUC = %0.2f)' % (i + 1, roc_auc))

        i += 1

    # Plot mean ROC Curve
    ax.plot([0, 1], [0, 1],
            linestyle='--',
            lw=3,
            color='k',
            label='No Skill',
            alpha=.8)
    mean_tpr = mean(tprs_, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = std(aucs)
    ax.plot(mean_fpr,
            mean_tpr,
            color='navy',
            label=r'Mean ROC (AUC = %0.3f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=4)

    # calculates the standard deviation and fills the +-1 standard deviation
    # of the mean ROC curve
    std_tpr = std(tprs_, axis=0)
    tprs_upper = minimum(mean_tpr + std_tpr, 1)
    tprs_lower = maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr,
                    tprs_lower,
                    tprs_upper,
                    color='grey',
                    alpha=.2,
                    label=r'$\pm$ 1 Standard Deviation')

    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_title(title)
    ax.legend(loc='lower right', fontsize='xx-small')

    return ax


def calculate_statistics(cv_scores):
    """Returns mean and standard deviation of CV scores
    """
    from numpy import array

    not_scores = ['precisions', 'recalls', 'fprs', 'tprs', 'predictions']
    mean_scores = {}
    std_dev = {}
    for k, v in cv_scores.items():
        mean_scores[k] = []
        std_dev[k] = []
        for key, value in v.items():
            if key not in not_scores:
                mean_scores[k].append(array(value).mean())
                std_dev[k].append(array(value).std())
    return mean_scores, std_dev


def make_df_statistics(cv_results):
    """Return results from `calculate_statistics` into a DataFrame"""
    from pandas import DataFrame

    metrics = [
        'accuracy_score', 'precision_score', 'recall_score', 'specificity_score',
        'f1_score', 'f2_score', 'geometric_mean_score', 'matthews_corrcoef',
        'roc_auc_score', 'pr_auc_score']
    new_metrics = metrics[-3:]
    df = DataFrame(cv_results)
    df['metrics'] = metrics
    df.set_index('metrics', inplace=True)
    df.index.name = None
    df = df.loc[new_metrics, :]
    df = df.T
    return df


def train_model(estimators, X, y, scalers=[False]):
    from sklearn.preprocessing import scale
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import fbeta_score
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import roc_curve

    scores = {}
    list_of_scoring_functions = scoring_functions()
    metrics = ['accuracy_score', 'precision_score', 'recall_score',
               'specificity_score', 'f1_score', 'f2_score',
               'geometric_mean_score', 'matthews_corrcoef', 'roc_auc_score',
               'pr_auc_score']

    for estimator, scaler in zip(estimators, scalers):
        ml_name, ml = estimator
        X_copy = X.copy()

        if scaler:
            X_scaled = scale(X_copy)
            ml.fit(X_scaled, y)
            preds = ml.predict(X_scaled)
            probas = ml.predict_proba(X_scaled)[:, 1]
        else:
            ml.fit(X, y)
            preds = ml.predict(X)
            probas = ml.predict_proba(X)[:, 1]

        if ml_name not in scores.keys():
            scores[ml_name] = {}

        for metric_name, metric in zip(metrics, list_of_scoring_functions):
            if metric_name not in scores[ml_name].keys():
                scores[ml_name][metric_name] = []
            if metric in [roc_auc_score, pr_auc_score]:
                scores[ml_name][metric_name].append(metric(y,
                                                           probas))
            elif metric == fbeta_score:
                scores[ml_name][metric_name].append(metric(y,
                                                           preds,
                                                           beta=2))
            else:
                scores[ml_name][metric_name].append(metric(y,
                                                           preds))
    return scores


def test_model(estimators, X, y, scalers=[False]):
    from sklearn.preprocessing import scale
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import fbeta_score
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import roc_curve

    scores = {}
    list_of_scoring_functions = scoring_functions()
    metrics = ['accuracy_score', 'precision_score', 'recall_score',
               'specificity_score', 'f1_score', 'f2_score',
               'geometric_mean_score', 'matthews_corrcoef', 'roc_auc_score',
               'pr_auc_score']

    for estimator, scaler in zip(estimators, scalers):
        ml_name, ml = estimator

        X_copy = X.copy()
        if scaler:
            X_scaled = scale(X_copy)
            preds = ml.predict(X_scaled)
            probas = ml.predict_proba(X_scaled)[:, 1]
        else:
            preds = ml.predict(X)
            probas = ml.predict_proba(X)[:, 1]

        precision, recall, threshold = precision_recall_curve(
            y, probas)
        fpr, tpr, threshold = roc_curve(y, probas)

        if ml_name not in scores.keys():
            scores[ml_name] = {}

        keys = ['precisions', 'recalls', 'fprs', 'tprs', 'predictions']
        values = [precision, recall, fpr, tpr, preds]
        for key, value in zip(keys, values):
            if key not in scores[ml_name].keys():
                scores[ml_name][key] = []
            scores[ml_name][key].append(value)

        for metric_name, metric in zip(metrics, list_of_scoring_functions):
            if metric_name not in scores[ml_name].keys():
                scores[ml_name][metric_name] = []
            if metric in [roc_auc_score, pr_auc_score]:
                scores[ml_name][metric_name].append(metric(y,
                                                           probas))
            elif metric == fbeta_score:
                scores[ml_name][metric_name].append(metric(y,
                                                           preds,
                                                           beta=2))
            else:
                scores[ml_name][metric_name].append(metric(y,
                                                           preds))
    return scores
