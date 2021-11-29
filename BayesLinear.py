import math
import numpy as np
import torch
from torch.nn import Module, Parameter
import torch.nn.init as init
import torch.nn.functional as F

class BayesLinear(Module):

    def __init__(self, prior_mu, prior_sigma, in_features, out_features, bias=True):
        super(BayesLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.prior_log_sigma = math.log(prior_sigma)
        
        self.weight_mu = Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_sigma = Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_eps', None)
        
        self.bias_mu = Parameter(torch.Tensor(out_features))
        self.bias_log_sigma = Parameter(torch.Tensor(out_features))
        self.register_buffer('bias_eps', None)
        self.bias = True
            
        self.bias_mu = Parameter(torch.Tensor(out_features))
        self.bias_log_sigma = Parameter(torch.Tensor(out_features))
        self.register_buffer('bias_eps', None)
        
        # Sample to assign values to the empty tensors
        self.reset_parameters()

    def reset_parameters(self):
        # Initialization method of Adv-BNN
        stdv = 1. / math.sqrt(self.weight_mu.size(1))
        #self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_mu.data.normal_(0, stdv)
        self.weight_log_sigma.data.fill_(self.prior_log_sigma)
        if self.bias :
            #self.bias_mu.data.uniform_(-stdv, stdv)
            self.bias_mu.data.normal_(0, stdv)
            self.bias_log_sigma.data.fill_(self.prior_log_sigma)
            
#             self.bias_log_sigma.data.fill_(self.prior_log_sigma)
    def forward(self, input):
        
        if self.weight_eps is None :
            weight = self.weight_mu + torch.exp(self.weight_log_sigma) * torch.randn_like(self.weight_log_sigma)
        else :
            weight = self.weight_mu + torch.exp(self.weight_log_sigma) * self.weight_eps
        
        if self.bias:
            if self.bias_eps is None :
                bias = self.bias_mu + torch.exp(self.bias_log_sigma) * torch.randn_like(self.bias_log_sigma)
            else :
                bias = self.bias_mu + torch.exp(self.bias_log_sigma) * self.bias_eps                
        else :
            bias = None
            
        return F.linear(input, weight, bias)

    def extra_repr(self):
       return 'prior_mu={}, prior_sigma={}, in_features={}, out_features={}, bias={}'.format(self.prior_mu, self.prior_sigma, self.in_features, self.out_features, self.bias is not None)

class BLSS(Module):

    def __init__(self, rho, prior_mu, prior_sigma, in_features, out_features, bias=True):
        super(BLSS, self).__init__()
        self.rho = rho
        self.in_features = in_features
        self.out_features = out_features

        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.prior_log_sigma = math.log(prior_sigma)

        self.weight_mu = Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_sigma = Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_eps', None)

        self.bias_mu = Parameter(torch.Tensor(out_features))
        self.bias_log_sigma = Parameter(torch.Tensor(out_features))
        self.register_buffer('bias_eps', None)
        self.bias = True

        self.bias_mu = Parameter(torch.Tensor(out_features))
        self.bias_log_sigma = Parameter(torch.Tensor(out_features))
        self.register_buffer('bias_eps', None)

        # Sample to assign values to the empty tensors
        self.reset_parameters()

    def reset_parameters(self):
        # Initialization method of Adv-BNN
        stdv = 1. / math.sqrt(self.weight_mu.size(1))
        #self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_mu.data.normal_(0, stdv)
        self.weight_log_sigma.data.fill_(self.prior_log_sigma)
        if self.bias :
            #self.bias_mu.data.uniform_(-stdv, stdv)
            self.bias_mu.data.normal_(0, stdv)
            self.bias_log_sigma.data.fill_(self.prior_log_sigma)

    def forward(self, input):

        if self.weight_eps is None :
            z_weight = self.weight_mu + torch.exp(self.weight_log_sigma) * torch.randn_like(self.weight_log_sigma)
            weight = self.rho * z_weight + (1-self.rho) * self.weight_mu + torch.exp(self.weight_log_sigma) * np.sqrt(1-self.rho**2)*torch.randn_like(self.weight_log_sigma)
            
        else :
            z_weight = self.weight_mu + torch.exp(self.weight_log_sigma) * self.weight_eps
            weight = self.rho * z_weight + (1-self.rho) * self.weight_mu + torch.exp(self.weight_log_sigma) * np.sqrt(1-self.rho**2)*self.weight_eps
            
        if self.bias:
            if self.bias_eps is None :
                z_bias = self.bias_mu + torch.exp(self.bias_log_sigma) * torch.randn_like(self.bias_log_sigma)
                bias = self.rho * z_bias + (1-self.rho) * self.bias_mu + torch.exp(self.bias_log_sigma) * np.sqrt(1-self.rho**2)*torch.randn_like(self.bias_log_sigma)
            else :
                z_bias = self.bias_mu + torch.exp(self.bias_log_sigma) * self.bias_eps
                bias = self.rho * z_bias + (1-self.rho) * self.bias_mu + torch.exp(self.bias_log_sigma) * np.sqrt(1-self.rho**2)*self.bias_eps
        
        else :
            bias = None

        return F.linear(input, weight, bias)

    def extra_repr(self):
       return 'prior_mu={}, prior_sigma={}, in_features={}, out_features={}, bias={}'.format(self.prior_mu, self.prior_sigma, self.in_features, self.out_features, self.bias is not None) 
