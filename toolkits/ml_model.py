from sklearn.metrics import confusion_matrix, recall_score
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, roc_auc_score, make_scorer
from scipy import interp
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, SelectFwe, chi2, SelectFdr
from scipy.stats import mannwhitneyu
from skrebate import MultiSURF
from enum import Enum 
from sklearn.preprocessing import Imputer, StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import pdb
import numpy as np
import pandas as pd


from tensorflow.keras import losses, metrics, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ProgbarLogger
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, AlphaDropout, Dense, Dropout, Input 
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf




# Metrics Evaluation
def get_single_outerCV_scores(X_train, y_onehot_train, X_test, y_onehot_test, best_estimator):
    test_probas_ = best_estimator.fit(X_train, y_onehot_train).predict_proba(X_test)
    train_probas_ = best_estimator.predict_proba(X_train)
    
    # compute accuracy
    test_acc = (accuracy_score(y_onehot_test, test_probas_[:, 1] > 0.5 ))
    train_acc = (accuracy_score(y_onehot_train, train_probas_[:, 1] > 0.5 ))
    
    # computer sensitivity and specificity
    tn, fp, fn, tp = confusion_matrix(y_onehot_test, test_probas_[:, 1]>0.5).ravel()
    
    test_specificity = tn / float(tn+fp)
    test_sensitivity = recall_score(y_onehot_test, test_probas_[:, 1]>0.5) # tp / (tp + fn)

    
    # compute roc curve 
    test_fpr, test_tpr, test_thresholds = roc_curve(y_onehot_test, test_probas_[:, 1], drop_intermediate = False)
    
    test_auc = roc_auc_score(y_onehot_test, test_probas_[:, 1])
    train_auc = roc_auc_score(y_onehot_train, train_probas_[:, 1])
    
    print(tn, fp, fn, tp, test_specificity, test_sensitivity, test_acc, test_auc)
    
    return test_fpr, test_tpr, test_thresholds, test_auc, train_auc, test_acc, train_acc, test_sensitivity, test_specificity




def get_nestedCV_score(pipeline, search_space, X, y_onehot, inner_cv = 10, outer_cv = 5, collection = None):
    # Choose cross-validation techniques for the inner and outer loops,
    # independently of the dataset.
    # E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.

#     inner_cv = StratifiedKFold(n_splits = inner_cv, shuffle = True )
    # inner_cv = LeaveOneOut()
    outer_cv = StratifiedKFold(n_splits = outer_cv, shuffle = True )
    
    
    # data preprocessing
    # X = X.values
    lec = LabelEncoder()
    y_onehot = lec.fit_transform(y_onehot)

    # Non_nested parameter search and scoring
    clf = GridSearchCV(estimator=pipeline, param_grid=search_space, 
                       cv = inner_cv, verbose = False,
                       scoring='roc_auc', n_jobs=-1) # scoring=accuracy



    # Nested CV with parameter optimization

    test_fprs = []
    test_tprs = []
    test_aucs = []
    train_aucs = []
    test_accs = []
    train_accs = []
    test_sensitivities = []
    test_specificities = []
    best_params = []
    classifiers = []
    features = []
    collections = []
    best_estimators = []
    cv_results = []
    
    
    test_mean_fpr = np.linspace(0, 1, 100)
    
    def collection(clf):
        selection_method = list(clf.best_estimator_.named_steps)[0]
        
        if selection_method == 'SelectFromModel':
            feature_indices =  clf.best_estimator_.named_steps['SelectFromModel'].get_support(indices = True)
        elif selection_method == 'selectK':
            feature_indices =  clf.best_estimator_.named_steps['selectK'].get_support(indices = True)
        elif selection_method == 'ReliefF':
            ind =  clf.best_estimator_.named_steps['ReliefF'].top_features_
            n = clf.best_estimator_.named_steps['ReliefF'].n_features_to_select
            feature_indices = ind[0:n]
        else:
            feature_indices = None
            
        best_estimator = clf.best_estimator_
        best_param = clf.best_params_
        cv_result = clf.cv_results_
        classifier = clf

        return {'best_estimator': best_estimator, 
                'feature_indices': feature_indices, 
                'best_param': best_param,
                'cv_res':cv_result,
                'clf': classifier}
    
    for train, test in outer_cv.split(X, y_onehot):
        # grid search for best estimator from inner CV
        # import pdb; pdb.set_trace()
        clf.fit(X[train], y_onehot[train])
        col = collection(clf)
        best_estimator = col['best_estimator']
        print(best_estimator in best_estimators)
        
        # collection model info
        collections.append(col)
        features.append(col['feature_indices'])
        classifiers.append(col['clf'])
        best_params.append(col['best_param'])
        best_estimators.append(col['best_estimator'])
        cv_results.append(col['cv_res'])

        
        # get outer CV metric scores
#         best_estimator = collection(clf)['best_estimator']
        test_fpr, test_tpr, thresholds, test_auc, train_auc, test_acc, train_acc, test_sensitivity, test_specificity= get_single_outerCV_scores(X[train], y_onehot[train], X[test], y_onehot[test], best_estimator)
        
#         print(interp(test_mean_fpr, test_fpr, test_tpr))
        
        test_tprs.append(interp(test_mean_fpr, test_fpr, test_tpr))
        test_tprs[-1][0] = 0.0
        test_aucs.append(test_auc)
        train_aucs.append(train_auc)  # from inner cv
        test_accs.append(test_acc)    # from outer cv
        train_accs.append(train_acc)
        test_sensitivities.append(test_sensitivity)
        test_specificities.append(test_specificity)
        
#         classifiers.append(collection(clf)['clf'])
        
    
    test_mean_tpr = np.mean(test_tprs, axis=0)
#     print(test_mean_tpr)
    test_mean_tpr[-1] = 1.0
#     mean_auc = auc(mean_fpr, mean_tpr)
    test_mean_auc = np.mean(test_aucs)
    train_mean_auc = np.mean(train_aucs)
    test_mean_acc = np.mean(test_accs)
    train_mean_acc = np.mean(train_accs)
    test_mean_sensitivity = np.mean(test_sensitivities)
    test_mean_specificity = np.mean(test_specificities)
    
    outerCV_scores = pd.DataFrame({'test_acc': test_accs, 
                                  'train_acc': train_accs, 
                                  'test_auc': test_aucs, 
                                  'train_auc': train_aucs, 
                                  'test_sensitivity': test_sensitivities,
                                  'test_specificity': test_specificities
                                    })
    
    return {'outerCV_scores': outerCV_scores, 
            'test_mean_sensitivity':test_mean_sensitivity,
            'test_mean_specificity': test_mean_specificity,
            'test_mean_tpr': test_mean_tpr,
            'test_mean_fpr': test_mean_fpr, 
            'test_mean_auc': test_mean_auc,
            'test_mean_acc': test_mean_acc,
            'best_params': best_params,
            'best_estimators': best_estimators,
            'features': features, 
            'cv_results': cv_results,
            'classifiers': classifiers, 
            'collections': collections
           }





def get_repeated_nestedCV_score(pipeline, search_space, X, y_onehot, expID = 0, n = 10):
    repeated_sensitivity = []
    repeated_specificity = []
    repeated_acc = []
    repeated_auc = []
    repeated_classifier = []
    
    for i in range(n):
        sc = get_nestedCV_score(pipeline, search_space, X, y_onehot)
        repeated_sensitivity.append(sc['test_mean_sensitivity'])
        repeated_specificity.append(sc['test_mean_specificity'])
        repeated_acc.append(sc['test_mean_acc'])
        repeated_auc.append(sc['test_mean_auc'])
        repeated_classifier.append(sc['classifiers'])

    repeated_nestedCV_scores = pd.DataFrame({
          'repeated_sensitivity': repeated_sensitivity, 
          'repeated_specificity': repeated_specificity, 
          'repeated_acc': repeated_acc, 
          'repeated_auc': repeated_auc, 
            })
    repeated_nestedCV_scores['ExpID'] = 0
    
    aggs = {}
    
    def myFunc(x) : 
        alpha = 0.95
        p = ((1.0-alpha)/2.0) * 100
        lower = max(0.0, np.percentile(x, p))
        p = (alpha+((1.0-alpha)/2.0)) * 100
        upper = min(1.0, np.percentile(x, p))
        return "{np.mean(x):.3f} [{lower:.3f} ~ {upper:.3f}]".format(x)

    for f in repeated_nestedCV_scores.columns:
        aggs[f] = {"{f}[95% CI]".format(f): func for func in [myFunc]}
    

    # aggregation
    summary = repeated_nestedCV_scores.groupby('ExpID').agg(aggs)
    
    return summary, repeated_nestedCV_scores


# ####################################
# statistical test
def myStatTest(X, y_onehot):
    '''
    return F and p value from mannwhitneyu test
    '''
    F = []
    pval = []
    for i in range(X.shape[1]):
        a, b = mannwhitneyu(X[:, i][y_onehot == 0], X[:, i][y_onehot == 1], alternative = 'two-sided')
        F.append(a)
        pval.append(b)
    return -np.array(F), np.array(pval)    


# Model optimization
class HyperpSpace(Enum):
    gamma = [0.0001, 0.001, 0.01, 0.1, 1, 10] # np.logspace(-3, 1, 5) # 1e-4,
    degree = [2, 3, 4, 5, 6] # np.arange(2,7)
    k = [18, 19, 20, 21, 22, 23, 24, 25] # np.arange(18, 26)
    threshold = ['mean', '10*mean', '20*mean', '30*mean', '40*mean']
    c = [0.01, 0.1, 1, 10, 100, 1000, 10000,100000, 1000000]
    

class get_svm_pipeline(object):
    def __init__(self, kernel):
        self.kernel = kernel
    
    def __call__(self, feature_selection_method):
        # feature selection method
        if feature_selection_method == 'rf':
            feature_selection = ('SelectFromModel', SelectFromModel(RandomForestClassifier(n_jobs=-1, max_depth=10, n_estimators=15))) # , threshold='20*mean'
            
        elif  feature_selection_method == 'mystat':
            feature_selection = ('selectK', SelectKBest(myStatTest, k = 18))
            
        elif  feature_selection_method == 'reliefF':
            feature_selection = ('ReliefF', MultiSURF(n_features_to_select=50, n_jobs=-1))
        else: 
            print('method not valid')
        
        # pipeline
        pipeline = Pipeline([   
            feature_selection, 
            ('std_scalar', StandardScaler()),
            ("svm", SVC(kernel='linear', verbose = 200, probability = True, decision_function_shape='ovo'))    
        ])
        
        # search space
        search_space = [{'svm__kernel': ['linear', 'rbf', 'poly'], 
                             'svm__C': HyperpSpace.c.value, 
                             'svm__gamma': HyperpSpace.gamma.value}]
        
        return pipeline, search_space
        
        
# ROC curve
def ROCplot(repeatedCV_scores):
    
    test_tpr = repeatedCV_scores['test_tpr']
    test_fpr = repeatedCV_scores['test_fpr']
    
    mean_auc = repeatedCV_scores['scores']['test_auc'].mean()
    
       # plot diagnoal line
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Chance', alpha=.8)
    
    # plot mean ROC curve
    test_mean_tpr = np.mean(test_tpr, axis=0)
    test_mean_fpr = np.mean(test_fpr, axis=0)
    mean_fpr = np.linspace(0, 1, 100)
    
    plt.plot(test_mean_fpr, test_mean_tpr, color='b', #marker = '*',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, 0.1),
         lw=2, alpha=.8)
    
    
    # plot 95% confidence interval  
    alpha = 0.95
    p = ((1.0-alpha)/2.0) * 100
    tprs_lower = np.percentile(test_tpr, p, axis = 0)
    p = (alpha+((1.0-alpha)/2.0)) * 100
    tprs_upper = np.percentile(test_tpr, p, axis = 0)
    
    print(len(tprs_lower), len(tprs_upper))
    
#     std_tpr = np.std(tprs, axis=0)
#     tprs_upper = max(0.0, np.percentile(x, p))
#     tprs_lower = np.maximum(mean_tpr - 2.262 * std_tpr/np.sqrt(len(tprs)), 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'95% confidence intervals (CI)')
    


    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    plt.gcf().set_size_inches(10, 8)
    plt.savefig('ROC.pdf')
    plt.show()
        
        
# Deep learning        
class get_deep_learning_pipeline(object)        :
    def __init__(self):
        pass

    def create_fc_model(self, input_shape = (400, ),
                        architecture = [10, 10, 10], 
                        activation = 'relu', 
                        dropout_rate = None):
        
        # model architecture
        model = Sequential()
        
        model.add(Dense(architecture[0], input_shape = (input_shape, )))
        if dropout_rate: 
            model.add(Dropout(dropout_rate))
        
        for nodes in architecture[1::]:
            model.add(Dense(nodes, activation = activation))
            if dropout_rate:
                model.add(Dropout(dropout_rate))
        
        model.add(Dense(1, activation = 'sigmoid'))    

        # compile
        model.compile(optimizer = optimizers.Adam(learning_rate = 0.00001), 
                      loss = losses.binary_crossentropy,
                      metrics = ['accuracy', 'AUC']      
        )

        return model

        # input = tf.keras.Input(shape = (X.shape[1], ))
        # for nodes in self.architecture:
        #     h = tf.keras.layers.Dense(nodes, actiation = self.activation)(input)
        #     tf.keras.layers.Dropout(self.dropout_rate)(h)

    def __call__(self, input_shape, architecture, activation, dropout_rate = None) :

        # pipeline
        estimator = KerasClassifier(build_fn = self.create_fc_model, 
                                    input_shape = input_shape, 
                                    architecture = architecture, 
                                    activation = activation,
                                    dropout_rate = dropout_rate,
                                    batch_size = 128, 
                                    validation_split = 0.3, 
                                    epochs = 200, 
                                    verbose = 2)
        return estimator
        

