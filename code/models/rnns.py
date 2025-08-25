''' Here we build a library of manually coded RNN architectures, primarily so we can return gate activations.

Note that all architectures are coded for batch_first = True inputs of shape (n_batch, n_seq, n_input_dim). 

@charlesdgburns'''

## setup ##

import torch
import torch.nn as nn
import math
import numpy as np

## Global variables ##


## Top level agent ## 


class TinyRNN(nn.Module):
  def __init__(self, input_size:int=3, hidden_size:int=1, out_size:int=2,
               rnn_type = 'GRU',
               sparsity_lambda:float = 1e-2):
    super().__init__()
    self.I = input_size
    self.H = hidden_size
    self.O = out_size
    self.rnn_type = rnn_type
    if rnn_type == 'GRU':
      self.rnn = ManualGRU(self.I,self.H)
    elif rnn_type == 'LSTM':
      self.rnn = ManualLSTM(self.I,self.H)
    #elif model_type == 'NMRNN':
    #  self.rnn = ManualNMRNN(self.I,self.H)
    self.decoder = nn.Linear(self.H, self.O)
    self.L1 = sparsity_lambda

  def forward(self, inputs):
    hidden, _ = self.rnn(inputs)
    predictions = self.decoder(hidden)
    return predictions

  def compute_losses(self, predictions, targets):
    prediction_loss = nn.functional.cross_entropy(predictions, targets) #NB: this applies softmax itself
    #for sparsity we need to select the right weights to regularise
    sparsity_loss = 0
    for name, param in self.rnn.named_parameters():
      if 'bias' not in name:
        sparsity_loss += self.L1*torch.abs(param).sum()
    return prediction_loss, sparsity_loss



## Manual RNN architectures ##



class ManualGRU(nn.Module):
  '''Manual GRU coded to return gate activations dictionary
  Returns: (hidden_sequence, h_past) if return_gate_activations=False,
           (hidden_sequence, gate_activations) if -||- = True'''
  def __init__(self,input_size,hidden_size):
    super().__init__() #init nn.Module
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.sigmoid = torch.nn.Sigmoid()
    self.tanh = torch.nn.Tanh()

    self.W_from_in = nn.Parameter(torch.Tensor(input_size, hidden_size*3))
    self.W_from_h = nn.Parameter(torch.Tensor(hidden_size, hidden_size*3))
    self.bias = nn.Parameter(torch.Tensor(hidden_size*6))
    self.init_weights()

  def init_weights(self):
    stdv = 1.0 / math.sqrt(self.hidden_size)
    for weight in self.parameters():
        weight.data.uniform_(-stdv, stdv)

  def forward(self, inputs, init_states = None, 
              return_gate_activations=False):
    ''' inputs are a tensor of shape (batch_size, sequence_size, input_size)
        outputs are tensor of shape (batch_size, sequence_size, hidden_size)'''

    batch_size, sequence_size, _ = inputs.shape
    hidden_sequence = []
    if return_gate_activations:
      gate_activations = {'reset':[],'update':[]}
    if init_states is None:
      h_past = torch.zeros(batch_size, self.hidden_size).to(inputs.device) #(n_hidden,batch_size)
    else:
      h_past = init_states

    for t in range(sequence_size):
      x_past = inputs[:,t,:] #(n_batch,input_size)
      #for computational efficiency we do two matrix multiplications and then do indexing further down:
      from_input = x_past@self.W_from_in + self.bias[:3*self.hidden_size]
      from_input = from_input.view(batch_size,3, self.hidden_size) #(n_batch,3,n_hidden)
      from_hidden = h_past@self.W_from_h + self.bias[3*self.hidden_size:]
      from_hidden = from_hidden.view(batch_size, 3, self.hidden_size) #(n_batch,3, n_hidden)
      r_t = self.sigmoid(from_input[:,0]+from_hidden[:,0]) #(n_batch,n_hidden), ranging from 0 to 1
      z_t = self.sigmoid(from_input[:,1]+from_hidden[:,1]) #(n_batch,n_hidden), ranging from 0 to 1; must have n_hidden because it is multiplied with hidden_state later.
      if return_gate_activations:
        gate_activations['reset'].append(r_t.unsqueeze(0)) 
        gate_activations['update'].append(z_t.unsqueeze(0))
      n_t = self.tanh(from_input[:,2]+r_t*(from_hidden[:,2])).view(batch_size, self.hidden_size) #(n_batch,n_hidden)
      h_past = (1-z_t)*n_t + z_t*h_past #(n_batch,hidden_size) #NOTE h_past is tehnically h_t now, but in the next for-loop it will be h_past. ;)
      hidden_sequence.append(h_past.unsqueeze(0)) #appending (1,n_batch,n_hidden) to a big list.

    hidden_sequence = torch.cat(hidden_sequence, dim=0) #(n_sequence, n_batch, n_hidden) gather all inputs along the first dimenstion
    hidden_sequence = hidden_sequence.transpose(0, 1).contiguous() #reshape to batch first (n_batch,n_seq,n_hidden)
    
    if return_gate_activations:
        for gate_label, activations in gate_activations.items():
            gate_activations[gate_label] = torch.cat(activations, dim=0).transpose(0,1).contiguous() #(n_batch,n_seq,n_hidden)
        return hidden_sequence, gate_activations
    else:
      return hidden_sequence, h_past #this is standard in Pytorch, to output sequence of hidden states alongside most recent hidden state.


class ManualLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz, device='cpu'):
        super().__init__()
        self.device=device
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.W = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_sz * 4))
        self.init_weights()


    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv).to(self.device)

    def forward(self, x, init_states=None,
                return_gate_activations = False):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size() #let's get our shapes right
        HS = self.hidden_size
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device),
                        torch.zeros(bs, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states

        hidden_sequence = []#initialise lists for unit activations
        if return_gate_activations:
            gate_activations = {'input':[],'forget':[],'output':[], 'candidate':[]}
        for t in range(seq_sz):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.W + h_t @ self.U + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]), # input
                torch.sigmoid(gates[:, HS:HS*2]), # forget
                torch.tanh(gates[:, HS*2:HS*3]), # this integrates inputs and past hidden (so over short-term memory)
                torch.sigmoid(gates[:, HS*3:]), # output
            )
            c_t = f_t * c_t + i_t * g_t #here we integrate inputs/STM with long-term memoryu
            h_t = o_t * torch.tanh(c_t)
            hidden_sequence.append(h_t.unsqueeze(0))
            if return_gate_activations:
                for gate_label, activation in {'input':i_t,'forget':f_t,'output':o_t,'candidate':c_t}.items():
                    gate_activations[gate_label].append(activation.unsqueeze(0))
        hidden_sequence = torch.cat(hidden_sequence, dim=0) #(sequence, batch, n_hidden)
        hidden_sequence = hidden_sequence.transpose(0, 1).contiguous() #(batch, sequence, feature)
        if return_gate_activations:
            for gate_label, activations in gate_activations.items():
                gate_activations[gate_label] = torch.cat(activations, dim=0).transpose(0,1).contiguous() #(n_batch,n_seq,n_hidden)
            return hidden_sequence, gate_activations
        else:
            return hidden_sequence, (h_t, c_t)

