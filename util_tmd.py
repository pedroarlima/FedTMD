import const_tmd as const
from random import shuffle
from sklearn import preprocessing
import pandas as pd
import os

# return time in hours, minutes and seconds from number of rows
def time_for_rows_number(rows_number):
    tot_shape = round(rows_number, 2)
    tot_h = int((tot_shape / (60 / const.WINDOW_DIMENSION)) / 60)
    tot_m = int((tot_shape / (60 / const.WINDOW_DIMENSION)) - (tot_h * 60))
    tot_s = int((tot_shape * const.WINDOW_DIMENSION) - (tot_h * 60 * 60) - (tot_m * 60))
    return tot_h, tot_m, tot_s


# return the number of row to return to be balanced with min_windows
def to_num(row, min_windows):
    return round((float(min_windows) / 100.00) * row['percent'], 0)


def split_data(df, train_perc=0.8, cv_perc=0.0, test_perc=0.2):
    assert train_perc + cv_perc + test_perc == 1.0
    # create random list of indices
    N = len(df)
    l = list(range(N))
    shuffle(l)
    # get splitting indicies
    trainLen = int(N * train_perc)
    cvLen = int(N * cv_perc)
    testLen = int(N * test_perc)
    # get training, cv, and test sets
    training = df.iloc[l[:trainLen]]
    cv = df.iloc[l[trainLen:trainLen + cvLen]]
    test = df.iloc[l[trainLen + cvLen:]]
    return training, cv, test


def average(l):
    return sum(l) * 1.0 / len(l)


def fill_nan_with_mean_training(training=pd.DataFrame(), test=pd.DataFrame()):

    trainingFill = training.copy()  
    trainingFill = trainingFill.fillna(trainingFill.mean(numeric_only=True))
    trainingFill = trainingFill.fillna(0)

    testFill = test.copy()
    testFill = testFill.fillna(trainingFill.mean(numeric_only=True))
    testFill = testFill.fillna(0)

    return trainingFill, testFill


def scale_features(train, test):
    # build scaler to apply on training and test the same transformation
    scaler = preprocessing.StandardScaler().fit(train)
    train_features_scaled = scaler.transform(train)
    test_features_scaled = scaler.transform(test)
    return train_features_scaled, test_features_scaled


def get_sets_for_classification(df_train, df_test, features):
    train, test = fill_nan_with_mean_training(df_train, df_test)

    classes = []
    classes2string = {}
    classes2number = {}

 
    train_features = train[features].values
    train_classes = [classes2number[c] for c in train['target'].values]
    test_features = test[features].values
    test_classes = [classes2number[c] for c in test['target'].values]
    
    return train_features, train_classes, test_features, test_classes


def separate_TMDataset_InUsers():

    # CARREGA O DATAFRAME COM TODOS OS USUÁRIOS
    df_balanced = pd.read_csv('./TransportationData/_Dataset/dataset_balanced.csv', index_col=False)

    # Criar um diretório para armazenar os datasets separados por usuários
    if not os.path.exists("user_datasets"):
        os.makedirs("user_datasets")

    # Separar o dataset por usuários
    unique_users = df_balanced['user'].unique()
    for user in unique_users:
        user_dataset = df_balanced[df_balanced['user'] == user]
        user_dataset.to_csv(f"user_datasets/{user}_dataset.csv", index=False)

    print("Conjuntos de treinamento e teste foram salvos com sucesso!")