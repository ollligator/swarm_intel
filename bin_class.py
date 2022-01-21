import collections
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score,make_scorer
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import pyswarms as ps
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import cohen_kappa_score
import smote_variants as sv

from SMOTE_BFT import SMOTE_BFT
from model import model


def check_notnull(data):
  plt.figure(figsize=(15,5))
  plt.xticks(rotation=90)
  plt.ylabel('Number')
  plt.title('Non-Missing Values in columns within %d instances '%data.shape[0])
  plt.bar(data.columns,data.notnull().sum())


def plot_displot(data):
    fig = plt.figure(1, figsize=(20, 40))

    for i in range(len(data.columns)):
        fig.add_subplot(10, 5, i + 1)
        sns.histplot(data.iloc[i], kde=True)
        plt.axvline(data[data.columns[i]].mean(), c='green')
        plt.axvline(data[data.columns[i]].median(), c='blue')

def plot_scatter(data,x,y,target):
  fig = plt.figure(1, figsize=(8,5))
  sns.scatterplot(data=data, x=x, y=y, hue=target)
  plt.xlabel('ftr# {}'.format(x))
  plt.ylabel('ftr# {}'.format(y))
  plt.show()

def plot_class_dist(target_column):
  ax = target_column.value_counts().plot(kind='bar', figsize=(6, 4), fontsize=12, color='#6ca5ce')
  ax.set_title('Target class\n', size=20, pad=30)
  ax.set_ylabel('Number of samples', fontsize=12)
  for i in ax.patches:
      ax.text(i.get_x() + 0.19, i.get_height(), str(round(i.get_height(), 2)), fontsize=12)

def fill_missing_values(data,num_features,cat_features):
  for f in num_features:
    median = data[f].mean()
    data[f].fillna(median, inplace=True)
  for col in cat_features:
    most_frequent_category=data[col].mode()[0]
    data[col].fillna(most_frequent_category,inplace=True)

def encode_target(data, target):
  label_encoder = LabelEncoder()
  target_encoded = label_encoder.fit_transform(data[target])
  return target_encoded

def standardize_data(data, num_features):
  scaler = StandardScaler()
  scaled_data = scaler.fit_transform(data[num_features])
  return scaled_data

def transfrom_cat_features(data, cat_features):
  for c in cat_features:
      data = data.merge(pd.get_dummies(data[c], prefix=c), left_index=True, right_index=True)
  data.drop(cat_features,axis=1,inplace=True)

def split_data(data,target):
  X_train, X_test, y_train, y_test = train_test_split(data, target, stratify=target, test_size=0.33, random_state=42)
  return X_train, X_test, y_train, y_test

def choice(x): return int(x)

def uniform(x): return x

def loguniform(x): return 10**x


def ErrorDistribs(y_true, y_pred):
    return abs(y_true - y_pred) / y_true


def tpr_weight_function(y_true, y_predict):
    d = pd.DataFrame()
    d['prob'] = list(y_predict)
    d['y'] = list(y_true)
    d = d.sort_values(['prob'], ascending=[0])
    y = d.y
    PosAll = pd.Series(y).value_counts()[1]
    NegAll = pd.Series(y).value_counts()[0]
    pCumsum = d['y'].cumsum()
    nCumsum = np.arange(len(y)) - pCumsum + 1
    pCumsumPer = pCumsum / PosAll
    nCumsumPer = nCumsum / NegAll
    TR1 = pCumsumPer[abs(nCumsumPer - 0.001).idxmin()]
    TR2 = pCumsumPer[abs(nCumsumPer - 0.005).idxmin()]
    TR3 = pCumsumPer[abs(nCumsumPer - 0.01).idxmin()]
    return 0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3

def get_oversampler(oversamplers_dict, oversampler_num, proportion):
        if proportion == None:
            return oversamplers_dict[oversampler_num]()
        else:
            return oversamplers_dict[oversampler_num](proportion=proportion)

def get_filter(filters_dict, filter_num):
        return filters_dict[filter_num]

def preprocess(data,target, num_features,cat_features):
    #check_notnull(data.drop(target, 1))
    fill_missing_values(data.drop(target, 1), num_features, cat_features)
    data[target] = encode_target(data, target)

    data[num_features] = standardize_data(data, num_features)
    transfrom_cat_features(data, cat_features)
    return data

def get_optimized_model(X_train, y_train):
    param_dict = {}
    param_dict['SVC'] = {
        'C': [uniform, 1, 10],
        'kernel': [choice, 0, 4, ['linear', 'poly', 'rbf', 'sigmoid']],
        'coef0': [loguniform, -2, 2],
        'class_weight': [choice, 0, 2, ['balanced', None]]
    }
    param_dict['KNeighborsClassifier'] = {
        'weights': [choice, 0, 2, ['uniform', 'distance']],
        'n_neighbors': [choice, 0, 9, [2, 3, 4, 5, 6, 7, 8, 9, 10]]
    }

    model_dict = {}
    model_dict['SVC'] = SVC
    model_dict['KNeighborsClassifier'] = KNeighborsClassifier
    print(model_dict)

    #auc_scorer = make_scorer(tpr_weight_function)
    for model_name in ['SVC']:  # add 'KNeighborsClassifier'
        clf_model = model()
        clf_model.train(X_train, y_train, model_dict[model_name], param_dict[model_name])
        #clf_model.predict(X_test)

    print(clf_model.best_params)
    return clf_model

def plot_pca(X,y,title):
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)
    tmp = pd.DataFrame(data=principalComponents
                               , columns=['principal component 1', 'principal component 2'])

    tmp['target'] = y
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    label=('0','1')
    ax.scatter(tmp['principal component 1'], tmp['principal component 2'], c=tmp['target'])
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()
def print_eval_results(y_test,preds):
    print('Classification report:')
    print(classification_report(y_test, preds))
    print('Geometric mean:', geometric_mean_score(y_test, preds, average='weighted'))
    print('Geometric mean default:', geometric_mean_score(y_test, preds))
    print('AUPRC', average_precision_score(y_test, preds))
    print('Cohen Kappa', cohen_kappa_score(y_test, preds))
def main():
    filename = 'datasets/pima.csv'
    target = "Class"
    #target = 'diahnosis
    data = pd.read_csv(filename, header=0)
    cat_features = []
    num_features = ['Preg', 'Plas', 'Pres', 'Skin', 'Insu', 'Mass', 'Pedi', 'Age']

    data=preprocess(data,target,num_features,cat_features)
    X_train, X_test, y_train, y_test = split_data(data.drop(target, 1), data[target])
    print('Trainset shape', X_train.shape)
    print('Testset shape', X_test.shape)
    print(y_train.value_counts() / y_train.shape[0])
    print(y_test.value_counts() / y_test.shape[0])
    clf_model=get_optimized_model(X_train,y_train)

    #clf_model = SVC()
    #clf_model = SVC(C=7.63, kernel='sigmoid', coef0=2.37)
    #clf_model.fit(X_train, y_train)
    preds = clf_model.predict(X_test)

    print('Classification report:')
    print(classification_report(y_test, preds))
    print('Geometric mean:',geometric_mean_score(y_test, preds,average='weighted'))
    print('Geometric mean default:', geometric_mean_score(y_test, preds))
    print('AUPRC',average_precision_score(y_test, preds))
    print('Cohen Kappa',cohen_kappa_score(y_test, preds))


    oversamplers_dict = {1: sv.G_SMOTE, 2: sv.SMOTE, 3: sv.RWO_sampling, 4: sv.SPY, 5: sv.ANS, 6: sv.kmeans_SMOTE}
    oversampler = get_oversampler(oversamplers_dict, 2, 1)
    X_sample, y_sample = oversampler.sample(X_train.values, y_train.values)
    print('Train size: ', collections.Counter(y_train))
    print('Gen size: ', collections.Counter(y_sample))
    filters_dict = {
        1: sv.TomekLinkRemoval(),
        2: sv.kmeans_SMOTE(),
        3: sv.CondensedNearestNeighbors(),
        4: sv.CNNTomekLinks(),
        #5: SMOTE_BFT(X_train.to_numpy(), y_train.to_numpy(),alpha=0.05, kneighbors=7, pl_min=0.9, proportion_min=0.8, proportion=1),
    }
    plot_pca(X_sample, y_sample, 'After oversampling')

    filter = get_filter(filters_dict, 1)
    X_filtered, y_filtered = filter.remove_noise(X_sample, y_sample)
    plot_pca(X_filtered, y_filtered, 'After filtering')

    # filter=SMOTE_BFT(X_train.values, y_train.values,proportion=1, kneighbors=1,alpha=1,pl_min=1,proportion_min=1)
    # X_filtered, y_filtered = filter.resample()

    print('Filt size: ', collections.Counter(y_filtered))
    model = SVC(C=7.63, kernel='sigmoid', coef0=2.37)
    model.fit(X_sample, y_sample)
    preds = model.predict(X_test)
    print_eval_results(y_test,preds)


if __name__ == '__main__':
    main()