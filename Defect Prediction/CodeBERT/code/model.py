# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable


class Model(nn.Module):
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
    
        # Define dropout layer, dropout_probability is taken from args.
        self.dropout = nn.Dropout(args.dropout_probability)


    def forward(self, input_ids=None, labels=None):
        #logits = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]
        outputs=self.encoder(input_ids,attention_mask=input_ids.ne(1))[0]
        #print("Encoder output shape:", outputs.shape)  
        outputs = self.dropout(outputs)
        #print("After dropout shape:", outputs.shape)
        logits=outputs
        #print(logits.shape)
        prob = F.softmax(torch.tensor(logits), dim=-1)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob


    def forward_with_hidden_states(self, input_ids, labels = None):
        outputs = self.encoder(input_ids,attention_mask=input_ids.ne(1), output_hidden_states = True)
        logits = outputs.logits  

        # Apply dropout
        logits = self.dropout(logits)
        #logits=outputs
        prob = F.softmax(torch.tensor(logits), dim=-1)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob, outputs.hidden_states
        else:
            return prob,outputs.hidden_states

    def forward_with_logits(self, input_ids, labels = None):
        outputs=self.encoder(input_ids,attention_mask=input_ids.ne(1))[0]

        # Apply dropout
        outputs = self.dropout(outputs)
        logits=outputs
        prob = F.softmax(torch.tensor(logits), dim=-1)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits



"""import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss

    
    
class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
    
        # Define dropout layer, dropout_probability is taken from args.
        self.dropout = nn.Dropout(args.dropout_probability)

        
    def forward(self, input_ids=None,labels=None): 
        outputs=self.encoder(input_ids,attention_mask=input_ids.ne(1))[0]

        # Apply dropout
        outputs = self.dropout(outputs)


        logits=outputs
        prob=torch.sigmoid(logits)
        if labels is not None:
            labels=labels.float()
            loss=torch.log(prob[:,0]+1e-10)*labels+torch.log((1-prob)[:,0]+1e-10)*(1-labels)
            loss=-loss.mean()
            return loss,prob
        else:
            return prob

    def forward_with_logits(self, input_ids=None,labels=None): 
        outputs=self.encoder(input_ids,attention_mask=input_ids.ne(1))[0]

        # Apply dropout
        outputs = self.dropout(outputs)
        
        logits=outputs
        prob=torch.sigmoid(logits)
        prob = torch.cat([(1 - prob), prob], dim=-1)
        if labels is not None:
            labels=labels.float()
            loss=torch.log(prob[:,0]+1e-10)*labels+torch.log((1-prob)[:,0]+1e-10)*(1-labels)
            loss=-loss.mean()
            return loss,prob
        else:
            return prob
      

    
    def forward_with_hidden_states(self, input_ids, labels = None):
        outputs = self.encoder(input_ids,attention_mask=input_ids.ne(1), output_hidden_states = True)

        # Apply dropout
        logits = self.dropout(outputs.logits)

        prob=torch.sigmoid(logits)
        if labels is not None:
            labels=labels.float()
            loss=torch.log(prob[:,0]+1e-10)*labels+torch.log((1-prob)[:,0]+1e-10)*(1-labels)
            loss=-loss.mean()
            return loss,prob, outputs.hidden_states
        else:
            return prob, outputs.hidden_states"""
        
 