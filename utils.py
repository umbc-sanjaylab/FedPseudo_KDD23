import logging
import numpy as np
import pandas as pd
import torch
import random
from torch.utils.data import TensorDataset, DataLoader
from pycox.evaluation import EvalSurv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import h5py
from main import *


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

    
    
def load_datasets(dataset_file):
    datasets = defaultdict(dict)

    with h5py.File(dataset_file, 'r') as fp:
        for ds in fp:
            for array in fp[ds]:
                datasets[ds][array] = fp[ds][array][:]

    return datasets    
    
def one_hot_encoder(data, encode):
    """
    Attributes:
    ----------    
    data: A dataframe 
    encode: A list of columns in the dataframe to be one-hot-encoded
    
    Returns
    -------
    An one-hot-encoded dataframe
    """    
    data_encoded = data.copy()
    encoded = pd.get_dummies(data_encoded, prefix=encode, columns=encode, drop_first=False)
    return encoded    
    

################################################## IID data Partition ######################################################

def sample_iid(data, num_centers, seed_no):
    """
    Randomly split data evenly across each center
    Arguments:
    data -- combined data
    num_centers -- number of centers
    Returns:
    Dict with center_id : indices of data assigned to center
    """
    np.random.seed(seed_no)
    n = data.shape[0]
    idxs = np.random.permutation(n)
    batch_idxs = np.array_split(idxs, num_centers)
    dict_center_idxs = {i: batch_idxs[i] for i in range(num_centers)}
    return dict_center_idxs 

################################################################################################################################




################################################## Non-IID data Partition ######################################################   
def sample_by_quantiles(df, num_centers):
    """
    Randomly split data by quantiles
    Arguments:
    df -- combined data as dataframe
    num_centers -- number of centres 
    Returns:
    Dict with center_id : indices of data assigned to center
    """
    
    # labels are tuples, features are DFs
    data = np.array(df['time'])
    dict_center_idxs, all_idxs = {}, np.array([i for i in range(len(data))])
    quantile = 1 / num_centers
    previous_idxs = torch.zeros(len(data),dtype=torch.bool).numpy()

    for i in range(num_centers):
        if quantile > 1:
            ValueError
        cutoff = np.quantile(data,quantile)
        selected_idxs = data <= cutoff 
        idxs_in_quantile = selected_idxs & ~previous_idxs
        previous_idxs = previous_idxs | idxs_in_quantile
        dict_center_idxs[i] = all_idxs[idxs_in_quantile]
        quantile += 1 / num_centers 

    return dict_center_idxs      

def mergeDictionary(dict_1, dict_2):
    ds = [dict_1, dict_2]
    d = {}
    for k in dict_1.keys():
        d[k] = np.concatenate(list(d[k] for d in ds))
    return d 

def non_iid_skewed(data, num_centers, seed_no):
    np.random.seed(seed_no)
    random.seed(seed_no)
    n = data.shape[0]
    idxs = range(0, n)
    idxs1 = np.array(random.sample(range(0, n), int(n/num_centers)))
    idxs2=np.setdiff1d(idxs,idxs1)
    data1 = data.loc[data.index[list(idxs1)], :]
    data2 = data.loc[data.index[list(idxs2)], :]
    dict_center_idxs1=sample_iid(data1, num_centers, seed_no)
    dict_center_idxs2=sample_by_quantiles(data2, num_centers)

    dict_center_idxs=mergeDictionary(dict_center_idxs1, dict_center_idxs2)
    return dict_center_idxs     

################################################################################################################################    
    
    
    
################################################## Get Local Dataset ######################################################    

def get_local_dataset(data_directory, dataset, partition, n_parties, client_id, seed_no):
    cols_to_remove=['time','status']
    if dataset=='metabric':
        column_name=['time','status','MKI67','EGFR','PGR','ERBB2','hormone_treatment','radiotherapy','chemotherapy','ER_positive','age_at_diagnosis']
        std_list=['MKI67','EGFR','PGR','ERBB2','age_at_diagnosis']
        
        # Downloaded the dataset from https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data/metabric
        datasets=load_datasets(data_directory+dataset+'/metabric_IHC4_clinical_train_test.h5') 
        training_data=datasets['train']
        test_dataset=datasets['test']
        train_df=pd.DataFrame(np.concatenate([training_data['t'].reshape(-1,1), training_data['e'].reshape(-1,1),training_data['x']],axis=1))
        test_df=pd.DataFrame(np.concatenate([test_dataset['t'].reshape(-1,1), test_dataset['e'].reshape(-1,1), test_dataset['x']],axis=1))
        central_data=pd.concat([train_df, test_df]).reset_index(drop=True)
        central_data.columns=column_name
        
        if partition == "centralized":
            dataidxs=sample_iid(central_data, 1, seed_no)
        elif partition == "iid":
            dataidxs=sample_iid(central_data, n_parties, seed_no)
        elif partition == "non-iid":  
            dataidxs=non_iid_skewed(central_data, n_parties, seed_no)
        
        central_data=np.array(central_data, dtype='float32') 
        local_data=central_data[dataidxs[client_id]] 
        feature_list = [x for x in column_name if x not in cols_to_remove]
        
    elif dataset == 'support':
        column_name=['time','status','age', 'sex', 'race', 'num_comorbid','diabetes', 'dementia', 'cancer', 'mean_abp', 'hr','resp_rate', 'temp', 'wbc_count','serum_sodium', 'serum_creatinine']
        OHE_list=['race','cancer']
        std_list=['age', 'num_comorbid', 'mean_abp', 'hr','resp_rate', 'temp', 'wbc_count','serum_sodium', 'serum_creatinine']
        # Downloaded the dataset from https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data/support
        datasets=load_datasets(data_directory+dataset+'/support_train_test.h5') 
        training_data=datasets['train']
        test_dataset=datasets['test']
        train_df=pd.DataFrame(np.concatenate([training_data['t'].reshape(-1,1), training_data['e'].reshape(-1,1),training_data['x']],axis=1))
        test_df=pd.DataFrame(np.concatenate([test_dataset['t'].reshape(-1,1), test_dataset['e'].reshape(-1,1), test_dataset['x']],axis=1))
        centralized_data=pd.concat([train_df, test_df]).reset_index(drop=True) 
        centralized_data.columns=column_name
        central_data = one_hot_encoder(centralized_data, OHE_list)
        
        if partition == "centralized":
            dataidxs=sample_iid(central_data, 1, seed_no)
        elif partition == "iid":
            dataidxs=sample_iid(central_data, n_parties, seed_no)
        elif partition == "non-iid":  
            dataidxs=non_iid_skewed(central_data, n_parties, seed_no)
        
        new_col_name=central_data.columns
        central_data=np.array(central_data, dtype='float32') 
        local_data=central_data[dataidxs[client_id]]         
        feature_list = [x for x in new_col_name if x not in cols_to_remove]
        
    elif dataset == 'gbsg':
        column_name=['time','status','htreat', 'tumgrad', 'menostat', 'age', 'posnodal', 'prm', 'esm']
        OHE_list=['tumgrad']
        std_list=['age', 'posnodal', 'prm', 'esm']   
        # Downloaded the dataset from https://github.com/jaredleekatzman/DeepSurv/tree/master/experiments/data/gbsg
        datasets=load_datasets(data_directory+dataset+'/gbsg_cancer_train_test.h5') 
        training_data=datasets['train']
        test_dataset=datasets['test']
        train_df=pd.DataFrame(np.concatenate([training_data['t'].reshape(-1,1), training_data['e'].reshape(-1,1),training_data['x']],axis=1))
        test_df=pd.DataFrame(np.concatenate([test_dataset['t'].reshape(-1,1), test_dataset['e'].reshape(-1,1), test_dataset['x']],axis=1))
        centralized_data=pd.concat([train_df, test_df]).reset_index(drop=True) 
        centralized_data.columns=column_name
        central_data = one_hot_encoder(centralized_data, OHE_list)
        
        if partition == "centralized":
            dataidxs=sample_iid(central_data, 1, seed_no)
        elif partition == "iid":
            dataidxs=sample_iid(central_data, n_parties, seed_no)
        elif partition == "non-iid":  
            dataidxs=non_iid_skewed(central_data, n_parties, seed_no)
        
        new_col_name=central_data.columns
        central_data=np.array(central_data, dtype='float32') 
        local_data=central_data[dataidxs[client_id]]          
        feature_list = [x for x in new_col_name if x not in cols_to_remove]
        
        
    else:
        local_data==pd.read_csv(data_directory+dataset+'/data_{}.csv'.format(client_id+1))        
        feature_list = [x for x in local_data.columns if x not in cols_to_remove]

    ## Split local data into train and test set
    loc_train_data, loc_test_data = train_test_split(local_data, test_size=0.2, random_state=seed_no)
        
    ## Get covariates, event time and event status
    train_time=np.array(loc_train_data[:,0], dtype='float32')
    train_status=np.array(loc_train_data[:,1], dtype='float32')
    
    test_time=np.array(loc_test_data[:,0], dtype='float32')
    test_status=np.array(loc_test_data[:,1], dtype='float32')
    
    X_train=np.array(loc_train_data[:,2:], dtype='float32')
    X_test=np.array(loc_test_data[:,2:], dtype='float32')

    
    
    return X_train, train_time, train_status, X_test, test_time, test_status, feature_list

################################################################################################################################
    

    

################################################## Evaluation metrics ######################################################    
 
def Evaluation(model, X_test, test_time, test_status, evaltime):
    """
    Attributes:
    ----------    
    model: Trained Model 
    X_test: Covariates in test set
    test_status: Event status array for test set
    test_time: Observed times array for test set
    evaltime: Prespecified evaluation times
    
    Returns
    -------
    Time-dependent C-index and Integrated Brier Score 
    """        
    device = "cuda" if torch.cuda.is_available() else "cpu"    
    ################ Performance Measure on entire test data #####################   
    sp_test = model(torch.tensor(X_test).to(device)) ## Predict the survival probability for test set  
    surv=pd.DataFrame(np.transpose(sp_test.cpu().detach().numpy())) # Transpose the prediction array    
    surv=surv.set_index([evaltime])  ## extract the predictions at evaluation time points
    ev = EvalSurv(surv, test_time, test_status, censor_surv='km') ## EvalSurv function from pycox package. 
    #Setting censor_surv='km' means that we estimate the censoring distribution by Kaplan-Meier on the test set.    
    cindex=ev.concordance_td() ## time-dependent cindex
    brier=ev.integrated_brier_score(evaltime) ## integrated brier score
    return cindex, brier

################################################################################################################################




################################################## Concodance Index metrics ######################################################    

def Concordance(model, x, durations, events, evaltime):
    """
    Attributes:
    ----------    
    model: Model 
    x: Covariates
    durations: Event status array 
    events: Observed times array 
    evaltime: Prespecified evaluation times
    Returns
    -------
    Time-dependent C-index
    """    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x=x.to(device)
    surv = model(x)
    y_pred = pd.DataFrame(np.transpose(surv.cpu().detach().numpy()))
    y_pred=y_pred.set_index([evaltime])
    ev = EvalSurv(y_pred, durations, events, censor_surv='km')
    cindex = ev.concordance_td()
    return cindex

################################################################################################################################




################################################## Pseudo Value based Loss Function ##############################################
def pseudo_loss(output, target):
    loss = torch.mean(target*(1-2*output)+(output**2)) 
    return loss

################################################################################################################################


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches
    
    
    
def save_model(model, model_index, args):
    logger.info("saving local model-{}".format(model_index))
    with open(args.modeldir+"trained_local_model"+str(model_index), "wb") as f_:
        torch.save(model.state_dict(), f_)
    return

def load_model(model, model_index, device="cpu"):
    with open("trained_local_model"+str(model_index), "rb") as f_:
        model.load_state_dict(torch.load(f_))
    model.to(device)
    return model



