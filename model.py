import torch
import torch.nn as nn
import torch.nn.functional as F

  
############################# FedPDNN  #####################################    
class FedPDNN(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super().__init__()
        self.surv_net = nn.Sequential(
            nn.Linear(in_features, 128), nn.SELU(),nn.Dropout(p=dropout),       
            nn.Linear(128, 64), nn.SELU(),nn.Dropout(p=dropout),                      
            nn.Linear(64, 64), nn.SELU(),nn.Dropout(p=dropout),
            nn.Linear(64, 32), nn.SELU(),nn.Dropout(p=dropout),
            nn.Linear(32, 32), nn.SELU(),nn.Dropout(p=dropout),
            nn.Linear(32, out_features), nn.Sigmoid()
        )

    def forward(self, input):
        output = self.surv_net(input)
        return output    
     
    
########################################## FedPLSTM #################################################

class FedPLSTM(nn.Module):
    def __init__(self, out_features, input_size, hidden_size, num_layers):
        super(FedPLSTM, self).__init__()
        self.out_features = out_features #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state


        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, hidden_size) #fully connected 1
        self.relu = nn.ReLU()        
        self.fc = nn.Linear(hidden_size, out_features) #fully connected last layer
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        x=torch.reshape(x, (-1, 1, x.shape[1]))
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) #hidden state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state        
        output = output[:, -1, :]
        out = self.fc_1(output) #first Dense
        out = self.relu(out) #relu 
        out = self.fc_1(out) #second Dense
        out = self.relu(out) #relu 
        out = self.fc(out) #Final Output
        out = self.sigmoid(out)
        return out
    
########################################## FedPAttn #################################################
    
class FedPAttn(nn.Module):
    def __init__(self, out_features, input_size, hidden_size, num_layers):
        super().__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True, bidirectional=True)
        self.out_features = out_features #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.fc_1 =  nn.Linear(2*hidden_size, 2*hidden_size) #fully connected 1
        self.relu = nn.ReLU()        
        self.fc = nn.Linear(2*hidden_size, out_features) #fully connected last layer
        self.sigmoid=nn.Sigmoid()

        
    def attention(self,x, input_size, hidden_size):
        """
            self attention : return self._z
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        i_q = torch.normal(0, 0.1, size=(self.input_size, self.hidden_size)).to(device)
        i_k = torch.normal(0, 0.1, size=(self.input_size, self.hidden_size)).to(device)
        i_v = torch.normal(0, 0.1, size=(self.input_size, self.hidden_size)).to(device)       
        i_w0 = torch.normal(0, 0.1, size=(self.hidden_size, self.input_size)).to(device)  
        
        q_trans = torch.tile(torch.reshape(i_q, [-1, self.input_size, self.hidden_size]), [x.shape[0], 1, 1])
        k_trans = torch.tile(torch.reshape(i_k, [-1, self.input_size, self.hidden_size]), [x.shape[0], 1, 1])
        v_trans = torch.tile(torch.reshape(i_v, [-1, self.input_size, self.hidden_size]), [x.shape[0], 1, 1])
        w0 = torch.tile(torch.reshape(i_w0, [-1, self.hidden_size, self.input_size]), [x.shape[0], 1, 1])
        q = torch.matmul(x, q_trans)
        k = torch.matmul(x, k_trans)
        v = torch.matmul(x, v_trans)  
        m = torch.nn.Softmax(dim=2)(torch.matmul(torch.matmul(q, torch.transpose(k, 2, 1))/8, v))
        z = torch.matmul(m, w0)
        z = torch.add(z,x)
        return z        


    def forward(self, x):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        x=torch.reshape(x, (-1, 1, x.shape[1])).to(device)
        attn_output = self.attention(x, self.input_size, self.hidden_size).clone().detach().to(device)
        h_0 = torch.zeros(2*self.num_layers, attn_output.size(0), self.hidden_size).to(device) #hidden state
        c_0 = torch.zeros(2*self.num_layers, attn_output.size(0), self.hidden_size).to(device) #internal state
        output, (hidden, cell) = self.rnn(attn_output, (h_0, c_0)) 
        out = self.fc(output) #Final Output
        out=torch.reshape(out, (-1, self.out_features))
        out = self.sigmoid(out)
        return out         
        
     