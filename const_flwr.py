import datetime
import tensorflow as tf
import util_flwr

current_time = datetime.datetime.now().strftime("%d_%m_%Y-%H_%M_%S")

#DIR_RESULTS = './FedTMD_Results_adam'
#DIR_RESULTS = './FedTMD_Results_sgd'
#DIR_RESULTS = './Results_FedTMD_adamgrad'
DIR_MODELS = './FedTMD_Clients_Models'

DIR_EVAL_COSTS_PER_ROUND = './FedTMD_eval_costs_per_round'
DIR_EVAL_COSTS_PER_ROUND_NO_SELECTION = './FedTMD_eval_costs_per_round_no_selection'

DIR_FIT_COSTS_PER_ROUND = './FedTMD_fit_costs_per_round'
DIR_FIT_COSTS_PER_ROUND_NO_SELECTION = './FedTMD_fit_costs_per_round_no_selection'

DIR_RESULTS_withSelection = './FedTMD_Results_sgd_with_selection'
DIR_RESULTS_withoutSelection = './FedTMD_Results_sgd_without_selection'

DIR_CONTR_RESULTS = './FedTMD_Results_contrSelection___'



FILE_AGGREGATED_EVALUATION_METRICS = 'aggregated_evaluation_metrics.csv'
FILE_AGGREGATED_FITTING_METRICS = 'aggregated_fit_metrics.csv'
     

NUM_ROUNDS = 1
LEARING_RATE = 0.1
BATCH_SIZE = 32
LOCAL_EPOCHS = 2
#LOCAL_EVALUATION_STEPS = 1


NUM_CLIENTS = 16
NUM_CLIENTS_SELECTION = (NUM_CLIENTS/2)
MOMENTUM_SGD = 0.0
ACTIVE_FN_FIRST_LAYER = tf.keras.activations.relu
ACTIVE_FN_LAST_LAYER = tf.keras.activations.softmax

NEURONS_FIRST_LAYER = 600
NEURONS_LAST_LAYER = 5

SENSOR_SET = 3
NUM_FEATURES = 36

# DIR_CONFIG_RESULTS = DIR_RESULTS + "/"+ str(NUM_ROUNDS) + "_" + str(LEARING_RATE) + "_" + str(BATCH_SIZE) + "_" + str(LOCAL_EPOCHS) + "/"

# DIR_CLIENTS_RESULTS = DIR_CONFIG_RESULTS + "/clients_results/"
# DIR_SERVER_RESULTS = DIR_CONFIG_RESULTS + "/server_results/"


AGGREGATED_FIT_DIR_WITH_SELECTION = DIR_RESULTS_withSelection + "/"+ str(NUM_ROUNDS) + "_" + str(LEARING_RATE) + "_" + str(BATCH_SIZE) + "_" + str(LOCAL_EPOCHS) + "/"
AGGREGATED_EVALUATOIN_DIR_WITH_SELECTION = DIR_RESULTS_withSelection + "/"+ str(NUM_ROUNDS) + "_" + str(LEARING_RATE) + "_" + str(BATCH_SIZE) + "_" + str(LOCAL_EPOCHS) + "/"

AGGREGATED_FIT_DIR_WITHOUT_SELECTION = DIR_RESULTS_withoutSelection + "/"+ str(NUM_ROUNDS) + "_" + str(LEARING_RATE) + "_" + str(BATCH_SIZE) + "_" + str(LOCAL_EPOCHS) + "/"
AGGREGATED_EVALUATOIN_DIR_WITHOUT_SELECTION = DIR_RESULTS_withoutSelection + "/"+ str(NUM_ROUNDS) + "_" + str(LEARING_RATE) + "_" + str(BATCH_SIZE) + "_" + str(LOCAL_EPOCHS) + "/"


#TENSORBOARD_SERVER_BASE_LOG_DIR = "./logs_simulation_tf" + "/" + str(NUM_ROUNDS) + "_" + str(LEARING_RATE) + "_" + str(BATCH_SIZE) + "_" + str(LOCAL_EPOCHS) + "/"
 

OPTIMIZER =  tf.keras.optimizers.Adam(learning_rate=LEARING_RATE)   #tf.keras.optimizers.SGD(learning_rate= LEARING_RATE, momentum=MOMENTUM_SGD)  #'sgd'   #SGD()   #tf.keras.optimizers.experimental.SGD()

LOSS_v1 = tf.keras.losses.SparseCategoricalCrossentropy()    
METRICS_v1 = [
    tf.keras.metrics.SparseCategoricalAccuracy()
]


LOSS_v2 = tf.keras.losses.CategoricalCrossentropy()
METRICS_v2 = [
    tf.keras.metrics.CategoricalAccuracy(),
    tf.keras.metrics.Precision(),
    tf.keras.metrics.Recall(),
    #F1Score(),
]