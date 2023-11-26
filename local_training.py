import numpy as np
import torch.optim as optim
import argparse
import logging
from model import *
from main import *
from utils import *

args = get_args()



## Function for training the model
def train(dataloader, model, loss_fn, optimizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_batches = len(dataloader) # Total number of observation divided by batch size
    model.train()
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device) # x is covariates and y is the pseudo values in the batch        
        # Compute prediction error
        pred = model(X)        
        loss = loss_fn(pred,y)
        # Backpropagation
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) 
    total_loss /= num_batches    
    return total_loss
        
## Function for evaluating the model
def evaluate(dataloader, model, loss_fn):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_batches = len(dataloader)
    model.eval()
    test_loss=0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device) # x is covarites and y is the pseudo values in the batch
            pred = model(X)
            test_loss += float(loss_fn(pred, y).item())
    test_loss /= num_batches

    return test_loss



#####################################  Training Models in "Centralized" setting  ########################################

def train_net_all(net_id, net, train_dataloader, val_dataloader, epochs, lr, args_optimizer, X_val, val_time, val_status, X_test, test_time, test_status, evaltime, device="cpu"):
    logger.info('Training network %s' % str(net_id))
    
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg) ## Default optimizer
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg, amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
   
    
    criterion = pseudo_loss    ## Pseudo value based loss function 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.patience) ## Learning rate scheduler
    Epochs = epochs       ## Number of local epochs
    patience=args.patience     ## Number of patience for early stopping
    best_val_cindex=0.0


    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_losses=[]
    val_losses=[]
    cindex=[]
    for epoch in range(Epochs):
        ## Calculate the train loss 
        train_loss = train(train_dataloader, net, criterion, optimizer)
        train_losses.append(train_loss)
        
        ## Calculate the validation loss 
        val_loss = evaluate(val_dataloader, net, criterion)
        scheduler.step(val_loss)
        val_losses.append(val_loss)

        
        ## Calculate the validation Cindex
        metrics=Concordance(net,torch.tensor(X_val),np.array(val_time),np.array(val_status), evaltime)
        cindex.append(metrics)

        ### Early Stopping Criteria
        if metrics > best_val_cindex:
            best_val_cindex = metrics
            es = 0
            torch.save(net.state_dict(), args.modeldir+'Centralized_{}_{}_{}_{}.pt'.format(args.model, args.dataset, args.init_seed, net_id))
        else:
            es += 1
            if es > patience:
                break
    
    
    ## Load the trained model and evaluate the performances
    net.load_state_dict(torch.load(args.modeldir+'Centralized_{}_{}_{}_{}.pt'.format(args.model, args.dataset, args.init_seed, net_id)))
    net.eval()
    test_cindex, test_brier = Evaluation(net, X_test, test_time, test_status, evaltime)  

    return test_cindex, test_brier



############################################################  Training Local Models  ##########################################################################

def train_net(net_id, net, train_dataloader, val_dataloader, epochs, lr, args_optimizer, X_val, val_time, val_status, X_test, test_time, test_status, evaltime, device="cpu"):
    logger.info('Training network %s' % str(net_id))
    
    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg) ## Default optimizer
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg, amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
   
    
    criterion = pseudo_loss    ## Pseudo value based loss function 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.patience) ## Learning rate scheduler
    Epochs = epochs       ## Number of local epochs
    patience=args.patience     ## Number of patience for early stopping
    best_val_cindex=0.0


    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_losses=[]
    val_losses=[]
    cindex=[]
    for epoch in range(Epochs):
        ## Calculate the train loss 
        train_loss = train(train_dataloader, net, criterion, optimizer)
        train_losses.append(train_loss)
        
        ## Calculate the validation loss 
        val_loss = evaluate(val_dataloader, net, criterion)
        scheduler.step(val_loss)
        val_losses.append(val_loss)

        
        ## Calculate the validation Cindex
        metrics=Concordance(net,torch.tensor(X_val),np.array(val_time),np.array(val_status), evaltime)
        cindex.append(metrics)

        ### Early Stopping Criteria
        if metrics > best_val_cindex:
            best_val_cindex = metrics
            es = 0
            torch.save(net.state_dict(), args.modeldir+'FedAvg_{}_{}_{}_{}.pt'.format(args.model, args.dataset, args.init_seed, net_id))
        else:
            es += 1
            if es > patience:
                break

    
    ## Load the trained model and evaluate the performances
    net.load_state_dict(torch.load(args.modeldir+'FedAvg_{}_{}_{}_{}.pt'.format(args.model, args.dataset, args.init_seed, net_id)))
    net.eval()    
    test_cindex, test_brier = Evaluation(net, X_test, test_time, test_status, evaltime)     
    return test_cindex, test_brier



def local_train_net(nets, selected, args, X_train_local, train_pseudo_local, train_time_local, train_status_local, X_val_local, val_pseudo_local, val_time_local, val_status_local, X_test_local, test_time_local, test_status_local, pseudo_evaltime, device="cpu"):
    
    avg_cindex = 0.0
    avg_brier=0.0
    for net_id, net in nets.items():
        if net_id not in selected:
            continue

        net.to(device)
        X_train_loc=X_train_local[net_id]
        train_pseudo_loc=train_pseudo_local[net_id]
        train_time_loc=train_time_local[net_id]
        train_status_loc=train_status_local[net_id]
        
        X_val_loc=X_val_local[net_id]
        val_pseudo_loc=val_pseudo_local[net_id]
        val_time_loc=val_time_local[net_id]
        val_status_loc=val_status_local[net_id]        
        
        
        X_test_loc=X_test_local[net_id]
        test_time_loc=test_time_local[net_id]  
        test_status_loc=test_status_local[net_id] 
        
        train_pseudo_loc=np.array(train_pseudo_loc, dtype='float32')
        val_pseudo_loc=np.array(val_pseudo_loc, dtype='float32')
        
 
        train_dl = FastTensorDataLoader(torch.from_numpy(X_train_loc), torch.from_numpy(train_pseudo_loc), batch_size=args.batch_size, shuffle=True)
        val_dl = FastTensorDataLoader(torch.from_numpy(X_val_loc), torch.from_numpy(val_pseudo_loc), batch_size=args.batch_size, shuffle=True)         

        test_cindex, test_brier = train_net(net_id, net, train_dl, val_dl, args.epochs, args.lr, args.optimizer, X_val_loc, val_time_loc, val_status_loc, X_test_loc, test_time_loc, test_status_loc, pseudo_evaltime, device=device)
        
        logger.info("net %d final test Cindex %f" % (net_id,test_cindex))
        logger.info("net %d final test Brier Score %f" % (net_id,test_brier))

        avg_cindex += test_cindex
        avg_brier += test_brier
        
    avg_cindex /= len(selected)
    avg_brier /= len(selected)

    
    if args.alg == 'local_training':
        logger.info("avg test Cindex %f" % (avg_cindex))
        logger.info("avg test Brier Score %f" % (avg_brier))

    nets_list = list(nets.values())
    return nets_list

#############################################################################################################################################################




