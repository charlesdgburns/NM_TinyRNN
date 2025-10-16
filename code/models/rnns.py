
''' Here we build a library of manually coded RNN architectures, primarily so we can return gate activations.

Note that all architectures are coded for batch_first = True inputs of shape (n_batch, n_seq, n_input_dim). 

@charlesdgburns'''

## setup ##

import torch
import torch.nn as nn
import math
import numpy as np

## Global variables ##
#here's an example dictionary we can pass TinyRNN with **OPTIONS_DICT
OPTIONS_DICT = {'rnn_type':'GRU',
                'input_size':3,
                'hidden_size':1,
                'out_size':2,
                'nm_size':1,
                'nm_dim':1,
                'nm_mode':'low_rank',
                'weight_seed':42,
                'sparsity_lambda':1e-4, #constrain weights (not biases)
                'energy_lambda':1e-2, #constrain hidden activations
                'input_forced_choice':False,
                'input_encoding': 'unipolar', #'unipolar' {0,1} or 'bipolar' {-1,1}
                'nonlinearity' :'tanh', # 'tanh' or 'relu'
                }

class TinyRNN(nn.Module):
  def __init__(self, 
               rnn_type = 'GRU',
               input_size:int = 3, 
               hidden_size:int = 1, 
               out_size:int = 2,
               nm_size: int = 1, #specific to NM_RNNs
               nm_dim: int = 1, #specific to NM_RNNs
               nm_mode: str = 'low_rank',
               sparsity_lambda:float = 1e-2,
               energy_lambda:float = 0,
               input_encoding = 'unipolar',
               input_forced_choice = False,
               nonlinearity = 'tanh',
               fixed_decoder = False,
               weight_seed = 42,
               batch_norm = False,
              ):
    super().__init__()
    
    # add attributes
    self.input_forced_choice = input_forced_choice
    self.input_size = input_size
    self.I = input_size if input_forced_choice else input_size-1
    self.H = hidden_size
    self.O = out_size
    self.rnn_type = rnn_type
    self.weight_seed = weight_seed
    self.sparsity_lambda = sparsity_lambda
    self.energy_lambda = energy_lambda
    self.nonlinearity = nonlinearity
    self.input_encoding = input_encoding
    self.fixed_decoder = fixed_decoder
    
    # We then need an RNN and a decoder:
    if rnn_type == 'vanilla':
      self.rnn = torch.nn.RNN(self.I,self.H, 
                              nonlinearity=self.nonlinearity) #vanilla RNN with tanh nonlinearity 
    elif rnn_type == 'GRU':
      self.rnn = ManualGRU(self.I,self.H, self.nonlinearity)
    elif rnn_type == 'LSTM':
      self.rnn = ManualLSTM(self.I,self.H, self.nonlinearity)
    elif rnn_type == 'monoGRU':
      self.rnn = MonoGated(self.I,self.H, self.nonlinearity)
    elif rnn_type == 'monoGRU2':
      self.rnn = MonoGated(self.I,self.H, self.nonlinearity, 
                           subnetwork = True)
    elif rnn_type == 'stereoGRU':
      self.rnn = StereoGated(self.I,self.H, self.nonlinearity)
    elif rnn_type == 'stereoGRU2':
      self.rnn = StereoGated(self.I,self.H, self.nonlinearity,
                             subnetwork = True)
    elif rnn_type == 'NMRNN':
      self.nm_size = nm_size
      self.nm_dim = nm_dim
      self.nm_mode = nm_mode
      self.rnn = ManualNMRNN(self.I,self.nm_size,self.nm_dim,self.H, self.nm_mode)
    
    self.decoder = nn.Linear(self.H, self.O)
    if batch_norm:
      self.batch_norm = nn.BatchNorm1d(self.I)
    # do a seeded  weight initialisation:
    self.init_weights()
    
  def init_weights(self):
    torch.manual_seed(self.weight_seed)
    np.random.seed(self.weight_seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed(self.weight_seed)
    stdv = 1e-3 ##1.0 / math.sqrt(self.H)
    for p in self.parameters():
        p.data.uniform_(-stdv, stdv)
    #if self.rnn_type == 'monoGRU':
      #self.rnn.W_hh.data = torch.tensor([[1.0e-3,0.0],
      #                                   [0.0,1.0e-3]])
      #self.decoder.weight.data = torch.tensor([[2.0,-2.0],
      #                                         [-2.0,2.0]])
      #self.rnn.bias_z.data = torch.tensor([1.0])
      
  def forward(self, inputs):
    '''Expects inputs shaped (n_batch, n_seq, n_features)
    For AB dataset, n_features are ordered as:
    'forced_choice','outcome','choice' coded as 0 or 1.'''
    if not self.input_forced_choice:
      inputs = inputs[:,:,1:]
    if self.input_encoding == 'bipolar':
      inputs = inputs*2-1 #maps 0 to -1 and 1 to 1.
    hidden, _ = self.rnn(inputs)
    if self.fixed_decoder:
      predictions = hidden @ torch.tensor([[2.0,-2.0],
                                           [-2.0,2.0]])
    else:
      predictions = self.decoder(hidden)
    return predictions, hidden

  def compute_losses(self, predictions,targets, hidden_states):
    ## predicion loss:
    prediction_loss = nn.functional.cross_entropy(predictions, targets) #NB: this applies softmax itself
    ## for weight sparsity we need to select the right weights to regularise:
    sparsity_loss = 0
    for name, param in self.rnn.named_parameters():
      if 'bias' not in name:
        sparsity_loss += torch.abs(param).sum()
    ## for energy loss we simply take mean squared activations:
    energy_loss = torch.mean(hidden_states**2)
    return prediction_loss, sparsity_loss*self.sparsity_lambda, energy_loss*self.energy_lambda
    
  def get_options_dict(self):
    '''Helper function to later save and reinstate model'''
    options_dict = {'rnn_type':self.rnn_type,
                    'input_size':self.input_size,
                    'hidden_size':self.H,
                    'out_size':self.O,
                    'sparsity_lambda':self.sparsity_lambda,
                    'energy_lambda': self.energy_lambda,
                    'weight_seed':self.weight_seed,
                    'nonlinearity':self.nonlinearity,
                    'input_encoding':self.input_encoding,
                    'input_forced_choice':self.input_forced_choice,
                    'batch_norm': hasattr(self,'batch_norm'),
                    'fixed_decoder':self.fixed_decoder}
    if self.rnn_type == 'NMRNN':
      options_dict['nm_size']=self.nm_size
      options_dict['nm_dim']=self.nm_dim
      options_dict['nm_mode']=self.nm_mode
    return options_dict
  
  def get_model_id(self):
    '''Helper function to save and reinstate model'''
    model_id = f'{self.H}_unit_{self.rnn_type}_{self.nonlinearity}_{self.input_encoding}'
    if self.rnn_type == 'NMRNN':
        model_id=f'{self.H}_unit_{self.rnn_type}_{self.nm_size}_subunits_{self.nm_dim}_{self.nm_mode}_{self.nonlinearity}_{self.input_encoding}'
    return model_id


## Custom minimal gated architectures ## 

class MonoGated(nn.Module):
  '''A minimal gated RNN with a 1D gating signal.'''
  def __init__(self, input_size,hidden_size, nonlinearity = 'tanh',
               subnetwork: bool = False):
    super().__init__() #init nn.Module
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.subnetwork = subnetwork
    self.sigmoid = nn.Sigmoid() #maybe consider setting this to something else
    if nonlinearity == 'relu':
      self.activation = nn.ReLU() 
    elif nonlinearity == 'tanh':
      self.activation = nn.Tanh()

    # Parameters
    self.W_ih = nn.Parameter(torch.Tensor(input_size, hidden_size))      # (I,H)
    self.W_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))     # (H,H)
    if self.subnetwork:
      self.W_z = nn.Parameter(torch.Tensor(hidden_size, 1))                # (I,Z)
      self.W_iz = nn.Parameter(torch.Tensor(input_size, hidden_size))                # (I,Z)
      self.W_hz = nn.Parameter(torch.Tensor(hidden_size, hidden_size))                # (I,Z)
    else:
      self.W_iz = nn.Parameter(torch.Tensor(input_size-1, 1))                # (I,Z)
      self.W_hz = nn.Parameter(torch.Tensor(hidden_size, 1))                # (I,Z)
    
    self.bias_h = nn.Parameter(torch.Tensor(hidden_size))                # (H,)
    self.bias_z = nn.Parameter(torch.Tensor(1))
  
  
  def forward(self, inputs, init_states = None,
              return_gate_activations = False,
              fixed_gates = {}):
    ''' inputs are a tensor of shape (batch_size, sequence_size, input_size)
        outputs are tensor of shape (batch_size, sequence_size, hidden_size)
        -----
        option to add fixed_gates dictionary with '_t' or 'z_t' keys, 
        which must be tensors of shape (n_batch,n_hidden).'''
    batch_size, sequence_size, _ = inputs.shape
    hidden_sequence = []
    if return_gate_activations:
      gate_activations = {'update':[]}
    if init_states is None:
      h_past = torch.zeros(batch_size, self.hidden_size).to(inputs.device) #(n_hidden,batch_size)
      z_past = torch.zeros(batch_size, self.hidden_size).to(inputs.device) #(n_hidden,batch_size)
    else:
      h_past, z_past = init_states
    
    for t in range(sequence_size):
      x_past = inputs[:,t,:] #(n_batch,input_size)
      #for computational efficiency we do two matrix multiplications and then do indexing further down:
      if self.subnetwork:
        z_past = self.activation(x_past@self.W_iz + z_past@self.W_hz) #(n_batch,n_hidden), ranging from 0 to 1; must have n_hidden because it is multiplied with hidden_state later.
        z_t = self.sigmoid(z_past@self.W_z+self.bias_z) #compress onto 1D
      else:
        z_t = self.sigmoid(x_past[:,0].unsqueeze(1)@self.W_iz + h_past@self.W_hz + self.bias_z)

      ## options to override gates and/or save them:
      if 'z_t' in fixed_gates.keys():
        z_t= fixed_gates['z_t']
      if return_gate_activations:
        gate_activations['update'].append(z_t.unsqueeze(0)) 
      ## continued computation of hidden state:
      n_t = self.activation(x_past@self.W_ih+h_past@self.W_hh+self.bias_h)
      h_past = (1-z_t)*n_t + (z_t)*h_past #(n_batch,hidden_size) #NOTE h_past is tehnically h_t now, but in the next for-loop it will be h_past. ;)
      hidden_sequence.append(h_past.unsqueeze(0)) #appending (1,n_batch,n_hidden) to a big list.

    hidden_sequence = torch.cat(hidden_sequence, dim=0) #(n_sequence, n_batch, n_hidden) gather all inputs along the first dimenstion
    hidden_sequence = hidden_sequence.transpose(0, 1).contiguous() #reshape to batch first (n_batch,n_seq,n_hidden)
    
    if return_gate_activations:
        for gate_label, activations in gate_activations.items():
            gate_activations[gate_label] = torch.cat(activations, dim=0).transpose(0,1).contiguous() #(n_batch,n_seq,n_hidden)
        return hidden_sequence, gate_activations
    else:
      return hidden_sequence, h_past #this is standard in Pytorch, to output sequence of hidden states alongside most recent hidden state.



class StereoGated(nn.Module):
  '''A minimal gated RNN with two 1D gating signals.'''
  def __init__(self, input_size,hidden_size, nonlinearity = 'tanh'):
    super().__init__() #init nn.Module
    self.input_size = input_size
    self.hidden_size = hidden_size
    
    self.sigmoid = nn.Sigmoid() #maybe consider setting this to something else
    if nonlinearity == 'relu':
      self.activation = nn.ReLU() 
    elif nonlinearity == 'tanh':
      self.activation = nn.Tanh()

    # Parameters
    self.W_ih = nn.Parameter(torch.Tensor(input_size, hidden_size))      # (I,H)
    self.W_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))     # (H,H)
    self.W_ir = nn.Parameter(torch.Tensor(input_size, 1))                # (I,Z)
    self.W_hr = nn.Parameter(torch.Tensor(input_size, 1))                # (I,Z)
    self.W_ii = nn.Parameter(torch.Tensor(input_size, 1))                # (I,Z)
    self.W_hi = nn.Parameter(torch.Tensor(input_size, 1))                # (I,Z)
    self.bias_h = nn.Parameter(torch.Tensor(hidden_size))                # (H,)
    self.bias_r = nn.Parameter(torch.Tensor(1))
    self.bias_i  = nn.Parameter(torch.Tensor(1))
    
  def forward(self, inputs, init_states = None,
              return_gate_activations = False,
              fixed_gates = {}):
    ''' inputs are a tensor of shape (batch_size, sequence_size, input_size)
        outputs are tensor of shape (batch_size, sequence_size, hidden_size)
        -----
        option to add fixed_gates dictionary with '_t' or 'z_t' keys, 
        which must be tensors of shape (n_batch,n_hidden).'''

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
      z_t = self.sigmoid(x_past@self.W_ir+h_past@self.W_hr +self.bias_r) #(n_batch,n_hidden), ranging from 0 to 1; must have n_hidden because it is multiplied with hidden_state later.
      r_t = self.sigmoid(x_past@self.W_ii+h_past@self.W_hi +self.bias_i) #(n_batch,n_hidden), ranging from 0 to 1; must have n_hidden because it is multiplied with hidden_state later.
      ## options to override gates and/or save them:
      if 'z_t' in fixed_gates.keys():
        z_t= fixed_gates['z_t']
      if 'r_t' in fixed_gates.keys():
        r_t = fixed_gates['r_t']
      if return_gate_activations:
        gate_activations['reset'].append(r_t.unsqueeze(0)) 
        gate_activations['update'].append(z_t.unsqueeze(0))
      ## continued computation of hidden state:
      n_t = self.activation(x_past@self.W_ih + (r_t*h_past)@self.W_hh+self.bias_h)
      h_past = z_t*h_past+(1-z_t)*n_t # (B,H) ##techincally h_t, but what was once past is now future
      
      hidden_sequence.append(h_past.unsqueeze(0)) #appending (1,n_batch,n_hidden) to a big list.
    hidden_sequence = torch.cat(hidden_sequence, dim=0) #(n_sequence, n_batch, n_hidden) gather all inputs along the first dimenstion
    hidden_sequence = hidden_sequence.transpose(0, 1).contiguous() #reshape to batch first (n_batch,n_seq,n_hidden)
    
    if return_gate_activations:
        for gate_label, activations in gate_activations.items():
            gate_activations[gate_label] = torch.cat(activations, dim=0).transpose(0,1).contiguous() #(n_batch,n_seq,n_hidden)
        return hidden_sequence, gate_activations
    else:
      return hidden_sequence, h_past #this is standard in Pytorch, to output sequence of hidden states alongside most recent hidden state.

## Manual RNN architectures ##

class ManualGRU(nn.Module):
  '''Manual GRU coded to return gate activations dictionary
  Returns: (hidden_sequence, h_past) if return_gate_activations=False,
           (hidden_sequence, gate_activations) if -||- = True'''
  def __init__(self,input_size,hidden_size, nonlinearity = 'tanh'):
    super().__init__() #init nn.Module
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.sigmoid = nn.Sigmoid()
    if nonlinearity == 'tanh':  
      self.activation = nn.Tanh()
    elif nonlinearity == 'relu':
      self.activation = nn.ReLU()

    self.W_from_in = nn.Parameter(torch.Tensor(input_size, hidden_size*3))
    self.W_from_h = nn.Parameter(torch.Tensor(hidden_size, hidden_size*3))
    self.bias = nn.Parameter(torch.Tensor(hidden_size*6))

  def forward(self, inputs, init_states = None, 
              return_gate_activations=False,
              fixed_gates = {}):
    ''' inputs are a tensor of shape (batch_size, sequence_size, input_size)
        outputs are tensor of shape (batch_size, sequence_size, hidden_size)
        -----
        option to add fixed_gates dictionary with 'r_t' or 'z_t' keys, 
        which must be tensors of shape (n_batch,n_hidden).'''

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
      ## options to override gates and/or save them:
      if 'z_t' in fixed_gates.keys():
        z_t= fixed_gates['z_t']
      if 'r_t' in fixed_gates.keys():
        r_t = fixed_gates['r_t']
      if return_gate_activations:
        gate_activations['reset'].append(r_t.unsqueeze(0)) 
        gate_activations['update'].append(z_t.unsqueeze(0))
      ## continued computation of hidden state:
      n_t = self.activation(from_input[:,2]+r_t*(from_hidden[:,2])).view(batch_size, self.hidden_size) #(n_batch,n_hidden)
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
    def __init__(self, input_sz, hidden_sz, nonlinearity = 'tanh', device='cpu'):
        super().__init__()
        self.device=device
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.W = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.U = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bias = nn.Parameter(torch.Tensor(hidden_sz * 4))
        if nonlinearity == 'tanh':
          self.activation = nn.Tanh()
        elif nonlinearity == 'relu':
          self.activation = nn.ReLU()
          
    def forward(self, x, init_states=None,
                return_gate_activations = False,
                fixed_gates = {}):
        """Assumes x is of shape (batch, sequence, feature),
        init_features is a tuple of (h_t,c_t) initial states
        ----
        option to include fixed_gates dict with 'f_t','i_t', 'g_t' or 'o_t' keys
        """
        bs, seq_sz, _ = x.size() #let's get our shapes right
        HS = self.hidden_size
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device),
                        torch.zeros(bs, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states

        hidden_sequence = [] #initialise lists for unit activations
        if return_gate_activations:
            gate_activations = {'input':[],'forget':[],'output':[], 'candidate':[]}
        for t in range(seq_sz):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.W + h_t @ self.U + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]), # input
                torch.sigmoid(gates[:, HS:HS*2]), # forget
                self.activation(gates[:, HS*2:HS*3]), # partial cell; this integrates inputs and past hidden (over short-term memory)
                torch.sigmoid(gates[:, HS*3:]), # output
            )
            ## option to fix certain gates:
            if 'i_t' in fixed_gates.keys():
              i_t = fixed_gates['i_t']
            if 'f_t' in fixed_gates.keys():
              f_t = fixed_gates['f_t']
            if 'g_t' in fixed_gates.keys():
              g_t = fixed_gates['g_t']
            if 'o_t' in fixed_gates.keys():
              o_t = fixed_gates['o_t']
            ## continued computation of candidate and hidden state:
            c_t = f_t * c_t + i_t * g_t #here we integrate inputs/STM with long-term memory
            h_t = o_t * self.activation(c_t)
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


class ManualNMRNN(nn.Module):
    def __init__(self, input_size, hidden_size,
                 nm_size, nm_dim, nm_mode='column',
                 nonlinearity = 'tanh',):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        if nonlinearity == 'tanh':  
          self.activation = nn.Tanh()
        elif nonlinearity == 'relu':
          self.activation = nn.ReLU()

       # assert nm_dim <= nm_size, "there must be at least as many subnetwork units as nm_dim size"
        self.nm_size = nm_size
        self.nm_dim = nm_dim
        self.hidden_size = hidden_size
        self.nm_mode = nm_mode  # 'low_rank', 'column', 'row'

        # Parameters
        self.W_ih = nn.Parameter(torch.Tensor(input_size, hidden_size))      # (I,H)
        self.W_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))     # (H,H)
        self.W_iz = nn.Parameter(torch.Tensor(input_size, nm_size))          # (I,Z)
        self.W_zz = nn.Parameter(torch.Tensor(nm_size, nm_size))             # (Z,Z)
        self.W_zk = nn.Parameter(torch.Tensor(nm_size, nm_dim))              # (Z,K)
        self.bias_z = nn.Parameter(torch.Tensor(nm_size))                    # (Z,)
        self.bias_h = nn.Parameter(torch.Tensor(hidden_size))                # (H,)
        self.bias_k = nn.Parameter(torch.Tensor(nm_dim))                     # (K,)

    def forward(self, inputs, init_states=None, 
                return_gate_activations= False,
                fixed_gates = {}):
        """
        inputs:  (B, T, I)
        returns: hidden_sequence (B, T, H), (h_T, z_T)
        ------
        option: fixed_gates dict with 'z_t' or 's_t' keys
        'z_t' must be of shape (B, nm_size), 's_t' shaped as (B,nm_dim)
        """
        device = inputs.device
        B, T, _ = inputs.size()
        H = self.hidden_size
        Z = self.nm_size
        K = self.nm_dim

        # Initial states
        if init_states is None:
            h_past = torch.zeros(B, H, device=device)
            z_past = torch.zeros(B, Z, device=device)   # <-- important: size nm_size
        else:
            h_past, z_past = init_states

        hidden_sequence = []
        if self.nm_mode == 'low_rank':
          U, S, Vh = torch.linalg.svd(self.W_hh, full_matrices=False)  # U:(H,H), S:(H,), Vh:(H,H)

        # TODO: could precompute low-rank decomposition here to speed thing up.
        # keeping separate for legibility for now
        if return_gate_activations:
          gate_activations = {'subnetwork':[],'modulation':[]}
        for t in range(T):
            x_t = inputs[:, t, :]                                   # (B,I)
            # Subnetwork dynamics -> modulation signal s_t
            z_t = self.activation(z_past @ self.W_zz + x_t @ self.W_iz +self.bias_z)   # (B, Z)
            s_t = self.sigmoid(z_t @ self.W_zk + self.bias_k)       # (B, K)
            # option to fix subnetwork states # 
            if 'z_t' in fixed_gates.keys():
              z_t = fixed_gates['z_t']
            if 's_t' in fixed_gates.keys():
              s_t = fixed_gates['s_t']
            # Build batched recurrent weight W_rec: (B,H,H)
            if self.nm_mode == 'low_rank':
                W_rec = self._make_Wrec_low_rank(s_t,U, S, Vh)               # (B,H,H)
            elif self.nm_mode == 'column':
                W_rec = self._make_Wrec_column(s_t)                 # (B,H,H)
            elif self.nm_mode == 'row':
                W_rec = self._make_Wrec_row(s_t)                    # (B,H,H)
            else:
                raise ValueError(f"Unknown nm_mode: {self.nm_mode}")
            # Recurrent step: h_t = tanh( h_{t-1} @ W_rec + x_t @ W_ih )
            # Use einsum for batched (B,H) × (B,H,H) -> (B,H)
            h_recur = torch.einsum('bi,bij->bj', h_past, W_rec)     # (B,H)
            h_t = self.activation(h_recur + x_t @ self.W_ih +self.bias_h) # (B,H)
            hidden_sequence.append(h_t.unsqueeze(1))                # (B,1,H)
            if return_gate_activations:
              gate_activations['subnetwork'].append(z_t.unsqueeze(1))
              gate_activations['modulation'].append(s_t.unsqueeze(1))
            # Update states
            h_past = h_t; z_past = z_t

        hidden_sequence = torch.cat(hidden_sequence, dim=1)         # (B,T,H)
        if return_gate_activations:
          for gate_label, activations in gate_activations.items():
              gate_activations[gate_label] = torch.cat(activations, dim=1) #(n_batch,n_seq,n_hidden)
          return hidden_sequence, gate_activations
        return hidden_sequence, (h_past, z_past)
        
    def _make_Wrec_low_rank(self, s_t, U,S,Vh):
        """
        Low-rank modulation
        s_t: (B, K)  where K = nm_dim
        Build W_rec[b] = sum_{k=1..K} s_t[b,k] * sigma_k * u_k v_k^T
        """
        H = self.hidden_size
        K = self.nm_dim
        assert K <= H, "nm_dim (number of components) cannot exceed hidden_size"

        # SVD once per forward pass (on CPU/GPU depending on parameter device)
        # take top-K components
        U_k = U[:, :K]                 # (H,K)
        S_k = S[:K]                    # (K,)
        Vh_k = Vh[:K, :]               # (K,H)
        # Prebuild the K rank-1 components C[k] = (S_k[k] * U_k[:,k]) ⊗ Vh_k[k,:]
        # L = (H,K), R = (K,H) -> C = (K,H,H)
        L = U_k * S_k.unsqueeze(0)     # (H,K) broadcast S_k across rows
        C = torch.einsum('ik,kj->kij', L, Vh_k)  # (K,H,H)
        # Combine per-batch with weights s_t: (B,K) × (K,H,H) -> (B,H,H)
        W_rec = torch.einsum('bk,kij->bij', s_t, C)
        return W_rec

    def _make_Wrec_column(self, s_t):
        """
        Column-wise modulation
        - If s_t has shape (B,1): global scalar per batch: W_rec[b] = s_t[b]*W_hh
        - If s_t has shape (B,H): per-column scaling: W_rec[b,:,j] = s_t[b,j] * W_hh[:,j]
        """
        B = s_t.shape[0]
        H = self.hidden_size
        W = self.W_hh.unsqueeze(0).expand(B, H, H)  # (B,H,H)
        if s_t.shape[1] == 1:
            return W * s_t.view(B, 1, 1) # NB: modulate Globally!
        elif s_t.shape[1] == H:
            return W * s_t.view(B, 1, H) # broadcast across rows -> scale columns
        else:
            raise ValueError(f"column mode expects nm_dim==1 or nm_dim==hidden_size ({H}), got {s_t.shape[1]}")

    def _make_Wrec_row(self, s_t):
        """
        Row-wise modulation
        - If s_t has shape (B,1): global scalar per batch: W_rec[b] = s_t[b]*W_hh
        - If s_t has shape (B,H): per-row scaling: W_rec[b,i,:] = s_t[b,i] * W_hh[i,:]
        """
        B = s_t.shape[0]
        H = self.hidden_size
        W = self.W_hh.unsqueeze(0).expand(B, H, H).clone()       # (B,H,H)
        if s_t.shape[1] == 1:
          W[:,0,:]*=s_t #modulate only the first row
          return W     
        elif s_t.shape[1] == H:
            return W * s_t.view(B, H, 1) # broadcast across cols -> scale rows
        else:
            raise ValueError(f"row mode expects nm_dim==1 or nm_dim==hidden_size ({H}), got {s_t.shape[1]}")

