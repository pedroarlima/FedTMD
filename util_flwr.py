from typing import Tuple, Union, List
import pickle
import os
import numpy as np
import pandas as pd
from scipy.sparse import spmatrix, issparse
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from flwr.common import Parameters
import tensorflow as tf
from sklearn.utils import resample

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
MLPParams = Union[List[np.ndarray], Tuple[List[np.ndarray], np.ndarray]]
XYList = List[XY]


def get_skl_model_weights(model: MLPClassifier) -> List[np.ndarray]:
    return [w for w in model.coefs_]


def get_skl_model_parameters(model: MLPClassifier):

    params = model.coefs_
    if len(model.intercepts_) > 0:
        params = (params, model.intercepts_)

    # Converter para NDArray
    params_array = np.concatenate([np.array(param, dtype=object).flatten() for param in params])
    
    return params_array


def set_skl_model_params(model: MLPClassifier, params: Parameters ) -> MLPClassifier:  #MLPParams
    
    # params_dict = {param_name: param_value for param_name, param_value in zip(model.get_params().keys(), params)}
    # model.set_params(params)
    
    model.coefs_ = params[0]
    if len(params) > 1:
        model.intercepts_ = params[1]
    return model


def set_initial_skl_params(model: MLPClassifier):
    """Sets initial parameters as zeros Required since model params are
    uninitialized until model.fit is called.

    But server asks for initial parameters from clients at launch. Refer
    to sklearn.neural_networks.MLPClassifier documentation for more
    information.
    """
    n_classes = 5 
    n_features = 36
    model.classes_ = np.array([i for i in range(n_classes)])

    model.coefs_ = np.zeros((n_classes, n_features))
    model.intercepts_ = np.zeros((n_classes,))


def load_TMDataset_tf(cid: str) :

    x_train_ = np.load(f'./user_datasets_flwr/U{cid}_datasets/x_train.npy', allow_pickle=True)
    x_train = np.array(x_train_)
    
    y_train = np.array(load_list_from_file(f'./user_datasets_flwr/U{cid}_datasets/y_train.pkl'), dtype=int)

    x_test_ = np.load(f'./user_datasets_flwr/U{cid}_datasets/x_test.npy', allow_pickle=True)
    x_test = np.array(x_test_)

    y_test = np.array(load_list_from_file(f'./user_datasets_flwr/U{cid}_datasets/y_test.pkl'), dtype=int)

    #print(f"DATASETS OF CLIENT {cid} LOADED")
    # print("X_TRAIN SIZE: ", len(x_train))
    # print("X_TEST SIZE: ", len(x_test))
           
    # unique_modes = len(np.unique(y_train))           
    # print("UNIQUE MODES: ", unique_modes)


    return x_train, y_train, x_test, y_test


def load_TMDataset_OneHot(cid: str) :

    x_train_ = np.load(f'./user_datasets_flwr/U{cid}_datasets/x_train.npy', allow_pickle=True)
    x_train = np.array(x_train_)
    
    y_train_ = np.array(load_list_from_file(f'./user_datasets_flwr/U{cid}_datasets/y_train.pkl'), dtype=int)

    x_test_ = np.load(f'./user_datasets_flwr/U{cid}_datasets/x_test.npy', allow_pickle=True)
    x_test = np.array(x_test_)

    y_test_ = np.array(load_list_from_file(f'./user_datasets_flwr/U{cid}_datasets/y_test.pkl'), dtype=int)

    # Formating dataset into one hot encoding 
    y_train = tf.keras.utils.to_categorical(y_train_, num_classes=5)
    y_test = tf.keras.utils.to_categorical(y_test_, num_classes=5)
    #print(f"DATASETS FOR CLIENT {cid} LOADED")

    return x_train, y_train, x_test, y_test


def load_TMDataset_OneHot_augmented(cid: str) :

    x_train_ = np.load(f'./user_datasets_flwr/U{cid}_datasets/x_train.npy', allow_pickle=True)
    x_train = np.array(x_train_)
    y_train_ = np.array(load_list_from_file(f'./user_datasets_flwr/U{cid}_datasets/y_train.pkl'), dtype=int)

    x_test_ = np.load(f'./user_datasets_flwr/U{cid}_datasets/x_test.npy', allow_pickle=True)
    x_test = np.array(x_test_)
    y_test_ = np.array(load_list_from_file(f'./user_datasets_flwr/U{cid}_datasets/y_test.pkl'), dtype=int)

    x_train_augmented, y_train_augmented = resample(x_train, y_train_, n_samples=10000, replace=True, random_state=42)
    x_test_augmented, y_test_augmented = resample(x_test, y_test_, n_samples=10000, replace=True, random_state=42)


    # Formating dataset into one hot encoding 
    y_train_augmented_final = tf.keras.utils.to_categorical(y_train_augmented, num_classes=5)
    y_test_augmented_final = tf.keras.utils.to_categorical(y_test_augmented, num_classes=5)
    print(f"AUGMENTED DATASETS FOR CLIENT {cid} LOADED")

    return x_train_augmented, y_train_augmented_final, x_test_augmented, y_test_augmented_final



def load_validation_dataset():

    x_val_ = np.load('x_train.npy', allow_pickle=True)
    x_val = np.array(x_val_)
    y_val_ = np.array(load_list_from_file('y_test.pkl'), dtype=int)
    y_val = y_val_[:-1]



    return x_val, y_val


def load_TMD_Centralized_dataset_OneHot():

    x_train_ = np.load('x_train.npy', allow_pickle=True)
    x_train = np.array(x_train_)
    
    y_train_ = np.array(load_list_from_file('y_train.pkl'), dtype=int)
    y_train = tf.keras.utils.to_categorical(y_train_, num_classes=5)    #y_val_[:-1]

    x_test_ = np.load('x_test.npy', allow_pickle=True)
    x_test = np.array(x_test_)

    y_test_ = np.array(load_list_from_file('y_test.pkl'), dtype=int)
    y_test = tf.keras.utils.to_categorical(y_test_, num_classes=5)

    return x_train, y_train, x_test, y_test


def resumo_antes(df_train: pd.DataFrame, df_test: pd.DataFrame):
    print("  **  ANALISES DATAFRAMES ANTES APLICAÇÃO FUNÇÕES GET_SET_FOR_CLASSIFICATION E SCALE_FEATURES  **  ")
        
    print("No ENTRADAS EM DF_TRAIN : ", len(df_train))
    print("RESUMO DE DF_TRAIN : ")
    resumo_df_train = df_train.describe()
    print(resumo_df_train)

    print("No ENTRADAS EM DF_TEST : ", len(df_test))
    print("RESUMO DE DF_TEST : ")
    resumo_df_test = df_test.describe()
    print(resumo_df_test)


def resumo_depois(x_train, y_train, x_test, y_test ):
    print("  **  ANALISES DATAFRAMES DEPOIS APLICAÇÃO FUNÇÕES GET_SET_FOR_CLASSIFICATION E SCALE_FEATURES  **  ")
    
    print("No ENTRADAS EM x_train : ", len(x_train))
    # Verificar se o array é esparsa (spmatrix)
    # if isinstance(x_train, spmatrix) or issparse(x_train):
        # Converter a matriz esparsa para um array compacto
    array = x_train.toarray()
    # Calcular as principais informações estatísticas
    resumo_xTrain = {
        'Mínimo': np.min(array),
        'Máximo': np.max(array),
        'Média': np.mean(array),
        'Mediana': np.median(array),
        'Desvio Padrão': np.std(array)
    }
    print(resumo_xTrain)

    print("No ENTRADAS EM y_train : ", len(y_train))
    # Converter a lista para um array numpy
    array = np.array(y_train)
    # Calcular as principais informações estatísticas
    resumo_yTrain = {
        'Mínimo': np.min(array),
        'Máximo': np.max(array),
        'Média': np.mean(array),
        'Mediana': np.median(array),
        'Desvio Padrão': np.std(array)
    }
    print(resumo_yTrain)

    print("No ENTRADAS EM x_test : ", len(x_test))
    # Verificar se o array é esparsa (spmatrix)
    # if isinstance(x_test, spmatrix) or issparse(x_test):
        # Converter a matriz esparsa para um array compacto
    array = x_test.toarray()
    # Calcular as principais informações estatísticas
    resumo_xTest = {
        'Mínimo': np.min(array),
        'Máximo': np.max(array),
        'Média': np.mean(array),
        'Mediana': np.median(array),
        'Desvio Padrão': np.std(array)
    }
    print(resumo_xTest)

    print("No ENTRADAS EM y_test : ", len(y_test))
    # Converter a lista para um array numpy
    array = np.array(y_test)
    # Calcular as principais informações estatísticas
    resumo_yTest = {
        'Mínimo': np.min(array),
        'Máximo': np.max(array),
        'Média': np.mean(array),
        'Mediana': np.median(array),
        'Desvio Padrão': np.std(array)
    }
    print(resumo_yTest)


def convert_to_dataframe(data):
    if isinstance(data, (list, np.ndarray)):
        df = pd.DataFrame(data)
    elif isinstance(data, spmatrix):
        df = pd.DataFrame(data.toarray())
    else:
        raise ValueError("Tipo de dados não suportado. Use uma lista, ndarray ou spmatrix.")
    
    return df


def save_list_to_file(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)


def load_list_from_file(filename) -> List :
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


def log_loss(y_true, y_pred):
    epsilon = 1e-15  # Valor pequeno para evitar log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Limita os valores de y_pred entre epsilon e 1 - epsilon
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))  # Cálculo da perda logarítmica
    return loss


def combine_data(x, y):

    # Verificar se o número de amostras em x_train é igual ao número de rótulos em y_train
    if len(x) != len(y):
        raise ValueError("O número de amostras em x_train é diferente do número de rótulos em y_train.")

    # Criar um DataFrame combinado sem índices
    combined_data = pd.DataFrame(x)
    combined_data['target'] = y
    combined_data.reset_index(drop=True, inplace=True)

    return combined_data


def cross_entropy_loss(y_true, y_pred):
    # Transforma os rótulos em formato de one-hot encoding
    num_classes = np.max(y_true) + 1
    y_true_one_hot = np.eye(num_classes)[np.array(y_true)]

    # Calcula a perda de entropia cruzada
    epsilon = 1e-15  # Adiciona uma pequena constante para evitar divisão por zero
    loss = -np.sum(y_true_one_hot * np.log(y_pred + epsilon)) / len(y_true)

    return loss


def custom_loss(y_true, y_pred):
    # Calcula a diferença absoluta entre as classes verdadeiras e as classes previstas
    loss = np.abs(y_true - y_pred)
    return loss


def split_dataset(user_id, csv_file, test_size=0.2, random_state=42):
    
    # Check if the directory exists
    if not os.path.exists(f'./user_datasets/U{user_id}_datasets'):
        os.mkdir(f'./user_datasets/U{user_id}_datasets')

    # Load the file CSV in a DataFrame
    # with open(csv_file, 'rb') as f:
    #     df = pd.read_csv(f) 
    # Carregar o arquivo CSV em um DataFrame   
    #df = pd.read_csv(csv_file)
    df = csv_file
    
    # Separar os recursos (x) e os rótulos (y)
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Dividir em conjunto de treinamento e teste
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    
    # Salvar os conjuntos de treinamento e teste em formato ndarray e list
    np.save(f'./user_datasets/U{user_id}_datasets/x_train.npy', x_train)
    np.save(f'./user_datasets/U{user_id}_datasets/x_test.npy', x_test)
    with open(f'./user_datasets/U{user_id}_datasets/y_train.pkl', 'wb') as f:
        pickle.dump(y_train, f)
    with open(f'./user_datasets/U{user_id}_datasets/y_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)
    
    print(f"Conjuntos de treinamento e teste do usuário {user_id} foram salvos com sucesso!")


def get_general_data_info():

    df_balanced = pd.read_csv('./TransportationData/_Dataset/dataset_balanced.csv', index_col=False)

    # Quantidade total de entradas
    total_entries = len(df_balanced)
    print("Quantidade total de entradas:", total_entries)

    # Quantidade total de features
    total_features = len(df_balanced.columns)
    print("Quantidade total de features:", total_features)

    # Obter a contagem de usuários únicos
    num_users = df_balanced['user'].nunique()
    # Exibir a quantidade de usuários únicos
    print("Quantidade de usuários únicos:", num_users)

    # Calcular a porcentagem de contribuição de cada usuário
    user_contributions = df_balanced['user'].value_counts(normalize=True) * 100
    print("Porcentagem de contribuição de cada usuário:")
    print(user_contributions)

    # df_UsersContributions = pd.DataFrame(user_contributions, columns=["proportion"])
    # df_UsersContributions.to_csv('df_UsersContributions.csv', index_label="user")


    # results_df = pd.DataFrame({
    #     'Metric': ['total_entries', 'total_features', 'num_users'],
    #     'Value': [total_entries, total_features, num_users]
    # })
    # results_df.to_csv('df_results__Analysing-TMDataset.csv', index=False)
    # print("RESULTS FILE STORAGED")



def set_num_clients_per_round(number):
    # Verificar se há um arquivo .csv de mesmo nome no diretório
    filename = './num_clients_file.csv'
    if not os.path.exists(filename):
        df = pd.DataFrame({'number': [number]})
        df.to_csv('./num_clients_file.csv', index=False)
    #else:
        # Crie um DataFrame com o valor e salve-o em um arquivo CSV
        # df = pd.DataFrame({'number': [number]})
        # df.to_csv('./num_clients_file.csv', index=False)
    
    return None

def get_num_client_per_round()-> int:
    df = pd.read_csv('./num_clients_file.csv')
    number = df['number'][0]
    return number


def set_parameters_size(size):
    # Verificar se há um arquivo .csv de mesmo nome no diretório
    filename = './parameters_size.csv'
    if not os.path.exists(filename):
        df = pd.DataFrame({'size': [size]})
        df.to_csv('./parameters_size.csv', index=False)
    #else:
        # # Crie um DataFrame com o valor e salve-o em um arquivo CSV
        # df = pd.DataFrame({'size': [size]})
        # df.to_csv('./parameters_size.csv', index=False)
    
    return None

def get_parameters_size()-> int:
    df = pd.read_csv('./parameters_size.csv')
    size = df['size'][0]
    return size


def set_num_rounds(rounds):
    # Verificar se há um arquivo .csv de mesmo nome no diretório
    file = './num_rounds.csv'
    if not os.path.exists(file):
        df = pd.DataFrame({'rounds': [rounds]})
        df.to_csv('./num_rounds.csv', index=False)
    else:
        os.remove(file)
        df = pd.DataFrame({'rounds': [rounds]})
        df.to_csv('./num_rounds.csv', index=False)


def get_num_rounds()-> int:
    df = pd.read_csv('./num_rounds.csv')
    rounds = df['rounds'][0]
    return rounds


def set_model_size(size):
    # Verificar se há um arquivo .csv de mesmo nome no diretório
    file = './model_size.csv'
    if not os.path.exists(file):
        df = pd.DataFrame({'size': [size]})
        df.to_csv('./model_size.csv', index=False)
    else:
        os.remove(file)
        df = pd.DataFrame({'size': [size]})
        df.to_csv('./model_size.csv', index=False)


def get_model_size()-> int:
    df = pd.read_csv('./model_size.csv')
    size = df['size'][0]
    return size