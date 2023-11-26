import os
import json
import numpy as np
import torch
import ray
import argparse
import logging
import copy
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from model import *
from local_training import *
from utils import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', 
                        type=str, 
                        default='FedPDNN', 
                        help='neural network used in training')
    parser.add_argument('--dataset', 
                        type=str, 
                        default='metabric', 
                        help='dataset used for training')
    parser.add_argument('--partition', 
                        type=str, 
                        default='iid', 
                        help='the data partitioning strategy')
    parser.add_argument('--batch-size', 
                        type=int, 
                        default=128, 
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', 
                        type=float, 
                        default=0.001, 
                        help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', 
                        type=int, 
                        default=20, 
                        help='number of local epochs')
    parser.add_argument('--n_parties', 
                        type=int, 
                        default=5,  
                        help='number of clients in the federated setting')
    parser.add_argument('--alg', 
                        type=str, 
                        default='fedavg',
                        help='communication strategy: fedavg/fedprox')
    parser.add_argument('--comm_round', 
                        type=int, 
                        default=20, 
                        help='number of maximum communication round')
    parser.add_argument('--patience', 
                        type=int, 
                        default=20, 
                        help='number of patience')
    parser.add_argument('--reg', 
                        type=float, 
                        default=1e-5, 
                        help="L2 regularization strength")
    parser.add_argument('--is_same_initial', 
                        type=int, 
                        default=1, 
                        help='Whether initial all the models with the same parameters in fedavg')
    parser.add_argument('--init_seed', 
                        type=int, 
                        default=0, 
                        help="Random seed")
    parser.add_argument('--log_file_name', 
                        type=str, 
                        default=None, 
                        help='The log file name')    
    parser.add_argument('--dropout_p', 
                        type=float, 
                        required=False, 
                        default=0.1, 
                        help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', 
                        type=str, 
                        required=False, 
                        default="./Data/", help="Data directory")
    parser.add_argument('--logdir', 
                        type=str, 
                        required=False, 
                        default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', 
                        type=str, 
                        required=False, 
                        default="./models/", help='Model directory path')
    parser.add_argument('--device', 
                        type=str, 
                        default='cuda:0', 
                        help='The device to run the program')
    parser.add_argument('--optimizer', 
                        type=str, 
                        default='adam', 
                        help='the optimizer')
    parser.add_argument('--sample', 
                        type=float, 
                        default=1, 
                        help='Sample ratio for each communication round')
    parser.add_argument('--pseudo_interval', 
                        type=int, 
                        default=10,  
                        help='Interval for pseudo values calculation') 
    parser.add_argument('--time_horizon_interval', 
                        type=float, 
                        default=1.0,  
                        help='Interval for survival function calculation') 
    parser.add_argument('--sensitivity', 
                        type=float, 
                        default=2.0, 
                        help='sensitivity parameter')
    parser.add_argument('--epsilon', 
                        type=float, 
                        default=5.5, 
                        help='privacy budget parameter')    
    parser.add_argument('--is_DP', 
                        default=False, 
                        action="store_true",
                        help='Enforcing DP in pseudo values')  
    
    args = parser.parse_args()
    return args

def init_nets(dropout_p, n_parties, args):

    nets = {net_i: None for net_i in range(n_parties)}

    for net_i in range(n_parties):

        if args.dataset == 'metabric':
            input_size  = 9
            output_size = 9
        elif args.dataset == 'support':
            input_size  = 25
            output_size = 9
        elif args.dataset == 'gbsg':
            input_size  = 9
            output_size = 9
        else:
            input_size  = 12
            output_size = 9
            
        if args.model == "FedPDNN":
            net = FedPDNN(input_size, output_size, dropout_p)
        elif args.model == "FedPLSTM":
            net = FedPLSTM(output_size, input_size, 128, 2)                      
        elif args.model == "FedPAttn":
            net = FedPAttn(output_size, input_size, 128, 2)   
              
        else:
            print("not supported yet")
            exit(1)
        nets[net_i] = net

    return nets

############################################################# Federated Pseudo Values Calculation ##########################################################



###################################################  Prespecified time points for pseudo values calculation  ###############################################
def global_unique_times():
    """
    Return:
        The unique times from all clients' unique event times
    """    
    local_unique_times=[]
    for client_id in range(args.n_parties):
        _, train_time, _, _, _, _,_=get_local_dataset(args.datadir, args.dataset, args.partition, args.n_parties, client_id, args.init_seed)
        local_unique_times.append(np.unique(train_time))
    glob_unique_time=np.unique(np.concatenate(local_unique_times))
    return glob_unique_time     

def Pseudo_Evaltime(global_unique_times):

    """
    Arguments:
        global_max_time: global unique times
        time_horizon_interval: Equal distant interval for dividing the time horizon
        pseudo_interval:interval for percentile of time at which pseudo values will be calculated 
    Return:
        A vector of the evaluation time points at which pseudo values will be calculated.
    """
    PseudoEvalTime=np.percentile(global_unique_times, np.arange(args.pseudo_interval, 99, args.pseudo_interval))
    return PseudoEvalTime 

## An alternative Approach:

# def global_max_time():
#     """
#     Return:
#         The maximum of the client's maximum event time      
#     """    
#     max_times=np.zeros((args.n_parties))
#     for client_id in range(args.n_parties):
#         _, train_time, _, _, _, _,_=get_local_dataset(args.datadir, args.dataset, args.partition, args.n_parties, client_id, args.init_seed)
#         max_times[client_id]=np.max(train_time)
#     glob_max_time=np.max(max_times)
#     return glob_max_time 

# def Pseudo_Evaltime(global_max_time):

#     """
#     Arguments:
#         global_max_time: maximum survival time
#         time_horizon_interval: Equal distant interval for dividing the time horizon
#         pseudo_interval:interval for percentile of time at which pseudo values will be calculated 
#     Return:
#         A vector of the evaluation time points at which pseudo values will be calculated.
#     """
#     PseudoEvalTime=np.percentile(np.arange(0.0, int(global_max_time), args.time_horizon_interval), np.arange(args.pseudo_interval, 99, args.pseudo_interval))
#     return PseudoEvalTime 


################################################################  Clients Execute  #######################################################################

# Compute the summary information in clients 

def compute_d_and_n(data, t):
    temp = data[data["time"] == t].groupby("status").count()
    try:
        d = temp.loc[1.0, "time"]
    except KeyError:
        d = 0
    try:
        c = temp.loc[0.0, "time"] 
    except KeyError:
        c = 0 
    return d, c


def compute_d_and_n_matrix(data):
    timeline = data["time"].sort_values().unique()
    di = np.full(len(timeline) + 1, np.nan)
    ci = np.full(len(timeline) + 1, np.nan)
    ni = np.full(len(timeline) + 1, 0)
    ni[0] = data.shape[0]
    for i in range(len(timeline)):
        d, c = compute_d_and_n(data, timeline[i])
        di[i] = d
        ci[i] = c
    m = pd.DataFrame(index=timeline)
    m["ni"] = ni[:-1]
    m["di"] = di[:-1]
    m["ci"] = ci[:-1]
    return m    

#local partial table calculation for each of the local clients

@ray.remote
def local_partial_matrix(client_time, client_status, sensitivity, epsilon, is_DP):
    """
    Arguments:
        client_time: Client's event time
        client_status: Client's event status
        sensitivity: sensitivity parameter
        epsilon: Privacy budget parameter. if epsilon is None it doesn't add Laplace noise, i.e., no differential privacy is enforced. If epsilon is provided, it adds Laplace noise, i.e., differential privacy is enforced.
        
    Return:
        A partial table with KM estimator components.
    """
    np.random.seed(args.init_seed)
    Surv_Eval_Time = sorted(np.unique(client_time))
    df = pd.concat([pd.DataFrame(client_time), pd.DataFrame(client_status)], axis=1)
    df.columns = ['time', 'status']
    local_partial_matrix = compute_d_and_n_matrix(df)
    local_partial_matrix.reset_index(inplace=True)
    local_partial_matrix = local_partial_matrix.rename(columns={'index': 'time'})
    local_partial_matrix.loc[1:,'ni']=0 

    if is_DP:    
        scale_parameter = sensitivity / epsilon
        Z = np.zeros((local_partial_matrix.shape[0], (local_partial_matrix.shape[1] - 1)))
        for i in range(local_partial_matrix.shape[0]):
            for j in range(local_partial_matrix.shape[1] - 1):
                Z[i, j] = np.random.laplace(loc=0, scale=scale_parameter)

        local_partial_matrix.iloc[:, 1:] = local_partial_matrix.iloc[:, 1:] + Z        
        local_partial_matrix[local_partial_matrix < 0] = 0
    
    return pd.DataFrame(local_partial_matrix)  


# local leave-one-out survival probabilities calculation for each client 
    
@ray.remote
def local_loo_survival_probability(global_partial_matrix, client_time, client_status):
    Surv_Eval_Time=np.array(global_partial_matrix.loc[:,'time'])
    local_loo_surv_prob=np.zeros((len(client_time),len(Surv_Eval_Time)))
    local_loo_matrix=np.zeros((len(client_time),len(Surv_Eval_Time),3))
    for i in range(len(client_time)):
        local_loo_matrix[i,0,0]=-1
        idx=np.where(client_time[i]==Surv_Eval_Time)
        if client_status[i]==1:
            local_loo_matrix[i,idx,1]=-1
        elif client_status[i]==0:
            local_loo_matrix[i,idx,2]=-1 
        local_loo_matrix[i,:,:]=np.array(global_partial_matrix)[:,1:]+local_loo_matrix[i,:,:]
        for p in range(len(Surv_Eval_Time)-1):
            local_loo_matrix[i,(p+1),0]=local_loo_matrix[i, (p),0]-(local_loo_matrix[i, (p),1]+local_loo_matrix[i,(p),2])  

    for j in range(len(client_time)):        
        local_loo_surv_prob[j,:]=np.cumprod(1-(local_loo_matrix[j,:,1]/(local_loo_matrix[j,:,0]+0.000000001)))

    return local_loo_surv_prob      

# Federated pseudo values calculation for each client at client side  

@ray.remote
def federated_pseudo_values(global_survival_probability, local_loo_survival_probability, Pseudo_Eval_Time, is_DP):
    local_loo_surv_probability = local_loo_survival_probability.copy()
    if is_DP:
        idx = np.where(global_survival_probability[1] == 0)
        local_loo_surv_probability[:, idx] = 0

    pseudo_values_all = global_survival_probability[0][0, 1] * global_survival_probability[1] - ((global_survival_probability[0][0, 1] - 1) * local_loo_surv_probability)

    index = np.searchsorted(np.array(global_survival_probability[0])[:, 0], Pseudo_Eval_Time, side='right') - 1
    pseudo_values = np.array(pd.DataFrame(pseudo_values_all).iloc[:, index])

    return pseudo_values    



###############################################################  Server Executes  ##########################################################################    
# global partial table calculation   
def global_partial_matrix(sensitivity, epsilon, is_DP):
    aggregated_local_partial_matrices=[]
    for client_id in range(args.n_parties):
        _, train_time, train_status, _, _, _,_=get_local_dataset(args.datadir, args.dataset, args.partition, args.n_parties, client_id, args.init_seed)
        aggregated_local_partial_matrices.append(local_partial_matrix.remote(train_time, train_status, sensitivity, epsilon, is_DP))
    aggregated_local_partial_matrices = ray.get(aggregated_local_partial_matrices)
    aggragate=pd.DataFrame(np.vstack(aggregated_local_partial_matrices))
    aggragate.columns=['time','nrisk','di','ci']  
    gp_matrix=aggragate.groupby('time', as_index=False).sum()
    gp_matrix.loc[0,'nrisk']=np.sum(gp_matrix.loc[:,'nrisk'])      
    gp_matrix.loc[1:,'nrisk']=0  
    return gp_matrix  


# global survival function calculation 
def global_survival_probability(global_partial_matrix, is_DP):
    global_full_matrix=np.array(global_partial_matrix.copy())
    r_last = global_full_matrix[0, 1]
    d_last = global_full_matrix[0, 2]
    c_last = global_full_matrix[0, 3]
    for j in range(1, global_full_matrix.shape[0]):
        r = r_last - d_last - c_last
        d = global_full_matrix[j, 2]
        c = global_full_matrix[j, 3]
        if r < 0:
            global_full_matrix[j, 1] = 0
        else:
            global_full_matrix[j, 1] = r
        if d < 0:
            global_full_matrix[j, 2] = 0
        if c < 0:
            global_full_matrix[j, 3] = 0        
        r_last = r
        d_last = d
        c_last = c

    if is_DP:
        indices = np.where(global_full_matrix[:, 1] <= global_full_matrix[:, 2])[0]        
        if indices.size == 0:
            global_full_matrix = global_full_matrix
        else:
            idx = indices[0]
            global_full_matrix[idx:, 1] = global_full_matrix[idx-1, 1]
            global_full_matrix[idx:, 2] = 0
    
        global_full_matrix[global_full_matrix < 0] = 0

    global_surv=np.cumprod(1-(global_full_matrix[:,2]/(global_full_matrix[:,1]+0.000000001))) # Calculate global survival probabilities using product rule formula
    return global_full_matrix, global_surv 
   



##############################################################################################################################################################
def load_data():
    
    # Compute pre-specified time for pseudo values, global partial table and global survival function
    pseudo_evaltime= Pseudo_Evaltime(global_unique_times()) 
    global_partial_table=global_partial_matrix(args.sensitivity, args.epsilon, args.is_DP)
    global_surv_prob=global_survival_probability(global_partial_table, args.is_DP)                     
                               

    # List of covariates to be standardized 
    if args.dataset == 'metabric':
        std_list=['MKI67','EGFR','PGR','ERBB2','age_at_diagnosis']
    elif args.dataset == 'support':
        std_list=['age', 'num_comorbid', 'mean_abp', 'hr','resp_rate', 'temp', 'wbc_count','serum_sodium', 'serum_creatinine']
    elif args.dataset == 'gbsg':
        std_list=['age', 'posnodal', 'prm', 'esm']        
    else:
        std_list=['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10','X11', 'X12']   
                         
    # Local Data Allocation                        
    X_train_local = []
    X_train_local_std = []
    train_pseudo_local = []
    train_time_local = []
    train_status_local = []     

    X_val_local = []
    X_val_local_std = []
    val_pseudo_local = []
    val_time_local = []
    val_status_local = [] 
                         
    X_test_local = []
    X_test_local_std = []
    test_time_local = []
    test_status_local = []       
    
    train_data_length = []   
    for client_id in range(args.n_parties):
        x_train_loc, train_time_loc, train_status_loc, X_test, test_time, test_status, feature_list=get_local_dataset(args.datadir, args.dataset, args.partition, args.n_parties, client_id, args.init_seed) 

        local_loo_surv= local_loo_survival_probability.remote(global_partial_table, train_time_loc, train_status_loc)
        pseudo_values=federated_pseudo_values.remote(global_surv_prob, local_loo_surv, pseudo_evaltime, args.is_DP)                          
        train_pseudo_loc=np.array(ray.get(pseudo_values), dtype='float32')                 
        X_train, X_val, train_pseudo, val_pseudo, train_time, val_time, train_status, val_status = train_test_split(x_train_loc, train_pseudo_loc, train_time_loc, train_status_loc, test_size=0.20, random_state=1234, stratify=train_status_loc)                  

        X_train_loc=pd.DataFrame(X_train)
        X_train_loc.columns=feature_list
        X_val_loc=pd.DataFrame(X_val)
        X_val_loc.columns=feature_list
        X_test_loc=pd.DataFrame(X_test)
        X_test_loc.columns=feature_list  

        scaler = StandardScaler().fit(X_train_loc.loc[:,std_list])
        X_train_loc.loc[:,std_list] =scaler.transform(X_train_loc.loc[:,std_list])
        X_val_loc.loc[:,std_list] = scaler.transform(X_val_loc.loc[:,std_list])
        X_test_loc.loc[:,std_list] = scaler.transform(X_test_loc.loc[:,std_list])

        X_train_loc=np.array(X_train_loc, dtype='float32')
        X_val_loc=np.array(X_val_loc, dtype='float32')
        X_test_loc=np.array(X_test_loc, dtype='float32')         

        X_train_central=np.array(X_train, dtype='float32')
        X_val_central=np.array(X_val, dtype='float32')
        X_test_central=np.array(X_test, dtype='float32') 
        
        ### Append the data from each of the clients
        X_train_local.append(X_train_central)
        X_train_local_std.append(X_train_loc)
        train_pseudo_local.append(train_pseudo)
        train_time_local.append(train_time)
        train_status_local.append(train_status)
        train_data_length.append(X_train.shape[0])
                         
        X_val_local.append(X_val_central)
        X_val_local_std.append(X_val_loc)
        val_pseudo_local.append(val_pseudo)
        val_time_local.append(val_time)
        val_status_local.append(val_status)                         
                         
        X_test_local.append(X_test_central)
        X_test_local_std.append(X_test_loc)
        test_time_local.append(test_time)
        test_status_local.append(test_status)                           
    
    X_train_centralized=np.vstack(X_train_local)
    X_train_centralized=np.array(X_train_centralized, dtype='float32')
    
    train_pseudo_global=np.vstack(train_pseudo_local)
    train_pseudo_global=np.array(train_pseudo_global, dtype='float32') 

    X_val_centralized=np.vstack(X_val_local)
    X_val_centralized=np.array(X_val_centralized, dtype='float32')
    
    val_pseudo_global=np.vstack(val_pseudo_local)
    val_pseudo_global=np.array(val_pseudo_global, dtype='float32') 
    
    val_time_global=np.concatenate(val_time_local)
    val_time_global=np.array(val_time_global, dtype='float32')    
    
    val_status_global=np.concatenate(val_status_local)
    val_status_global=np.array(val_status_global, dtype='float32')                          
    
    X_test_centralized=np.vstack(X_test_local)
    X_test_centralized=np.array(X_test_centralized, dtype='float32')
    
    X_test_global=np.vstack(X_test_local_std)
    X_test_global=np.array(X_test_global, dtype='float32')
        
    test_time_global=np.concatenate(test_time_local)
    test_time_global=np.array(test_time_global, dtype='float32')    
    
    test_status_global=np.concatenate(test_status_local)
    test_status_global=np.array(test_status_global, dtype='float32')   
                         
                         
    X_train_centralized=pd.DataFrame(X_train_centralized)
    X_train_centralized.columns=feature_list
    X_val_centralized=pd.DataFrame(X_val_centralized)
    X_val_centralized.columns=feature_list
    X_test_centralized=pd.DataFrame(X_test_centralized)
    X_test_centralized.columns=feature_list  
    
    # Standardize the continuous variables
    scaler = StandardScaler().fit(X_train_centralized.loc[:,std_list])
    X_train_centralized.loc[:,std_list] =scaler.transform(X_train_centralized.loc[:,std_list])
    X_val_centralized.loc[:,std_list] = scaler.transform(X_val_centralized.loc[:,std_list])
    X_test_centralized.loc[:,std_list] = scaler.transform(X_test_centralized.loc[:,std_list])

    X_train_centralized=np.array(X_train_centralized, dtype='float32')
    X_val_centralized=np.array(X_val_centralized, dtype='float32')
    X_test_centralized=np.array(X_test_centralized, dtype='float32') 
    
    X_test_global=np.array(X_test_global, dtype='float32') 

    
    # Create train and validation dataloader 
    central_train_dl = FastTensorDataLoader(torch.from_numpy(X_train_centralized), torch.from_numpy(train_pseudo_global), batch_size=args.batch_size, shuffle=True)
    central_val_dl = FastTensorDataLoader(torch.from_numpy(X_val_centralized), torch.from_numpy(val_pseudo_global), batch_size=args.batch_size, shuffle=True)          
    
    return central_train_dl, central_val_dl, X_val_centralized, val_time_global, val_status_global, X_test_centralized, X_test_global, test_time_global, test_status_global, X_train_local_std, train_pseudo_local, train_time_local, train_status_local, X_val_local_std, val_pseudo_local, val_time_local, val_status_local, X_test_local_std, test_time_local, test_status_local, train_data_length, pseudo_evaltime

##############################################################################################################################################################

                                        #Main: Centralized Training, Local Training, and Federated Training#

##############################################################################################################################################################

if __name__ == '__main__':
    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    if args.log_file_name is None:
        argument_path='experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
    else:
        argument_path=args.log_file_name+'.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device(args.device)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S"))
    log_path=args.log_file_name+'.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M',  filemode='w')
    logger = logging.getLogger()
    logger.info(device)

    
    import time
    start = time.time()     
    ray.init(ignore_reinit_error=True)
                
    
    device = "cuda" if torch.cuda.is_available() else "cpu"  
    central_train_dl, central_val_dl, X_val_centralized, val_time_global, val_status_global, X_test_centralized, X_test_global, test_time_global, test_status_global, X_train_local_std, train_pseudo_local, train_time_local, train_status_local, X_val_local_std, val_pseudo_local, val_time_local, val_status_local, X_test_local_std, test_time_local, test_status_local, train_data_length, pseudo_evaltime = load_data()
    ray.shutdown()
   
    
###########################################################  Centralized Training ###########################################################################          
    if args.alg == 'all_in':
        nets = init_nets(args.dropout_p, 1, args)
        
        test_cindex, test_brier = train_net_all(0, nets[0].to(device), central_train_dl, central_val_dl, args.epochs, args.lr, args.optimizer, X_val_centralized, val_time_global, val_status_global, X_test_centralized, test_time_global, test_status_global, pseudo_evaltime, device=device)

        logger.info('>> Centralized Test Cindex: {}'.format(test_cindex))
        logger.info('>> Centralized Test IBS: {}'.format(test_brier))

##############################################################################################################################################################

    
################################################################## Local Training ############################################################################                        
    elif args.alg == 'local_training':
        logger.info("Initializing nets")
        nets = init_nets(args.dropout_p, args.n_parties, args)
        arr = np.arange(args.n_parties)
        local_train_net(nets, arr, args, X_train_local_std, train_pseudo_local, train_time_local, train_status_local, X_val_local_std, val_pseudo_local, val_time_local, val_status_local, X_test_local_std, test_time_local, test_status_local, pseudo_evaltime, device=device)   
                         
##############################################################################################################################################################
    
    
############################################################ Federated Training: FedAvg ######################################################################        
    elif args.alg == 'fedavg':
        logger.info("Initializing nets")
        nets = init_nets(args.dropout_p, args.n_parties, args)
        global_models = init_nets(args.dropout_p, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))
            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]
            

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net(nets, selected, args, X_train_local_std, train_pseudo_local, train_time_local, train_status_local, X_val_local_std, val_pseudo_local, val_time_local, val_status_local, X_test_local_std, test_time_local, test_status_local, pseudo_evaltime, device=device)            

            # update global model
            total_data_points = sum([train_data_length[r] for r in selected])
            fed_avg_freqs = [train_data_length[r] / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)


            test_cindex, test_brier = Evaluation(global_model.to(device), X_test_global, test_time_global, test_status_global, pseudo_evaltime)
            
            logger.info('>> Global Model Test Cindex: {}'.format(test_cindex))
            logger.info('>> Global Model Test Brier: {}'.format(test_brier))

        
    # end time
    end = time.time()
    # total time taken
    print(f"Runtime of the {args.model} model is {end - start} sec")            
        
##############################################################################################################################################################
                
