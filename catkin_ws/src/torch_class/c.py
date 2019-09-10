#!/usr/bin/env python
# -*- coding: utf-8 -*-
#based on https://github.com/jcjohnson/pytorch-examples/blob/master/nn/two_layer_net_nn.py
# with some changes...

import torch
import gc
try:
    import cPickle as pickle
except:
    import pickle

"""
A fully-connected ReLU network with one hidden layer, trained to predict y from x
by minimizing squared Euclidean distance.

This implementation uses the nn package from PyTorch to build the network.
PyTorch autograd makes it easy to define computational graphs and take gradients,
but raw autograd can be a bit too low-level for defining complex neural networks;
this is where the nn package can help. The nn package defines a set of Modules,
which you can think of as a neural network layer that has produces output from
input and may have some trainable weights or other state.
"""
model = None
loss_fn = None
H = 5000
learning_rate = 0.005
num_epochs = 10
modeltype = '1-hidden'

def memstuff():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj),obj.size())

def savemodel(namename):
    global model
    f = open('%s.obj'%namename,'wb')
    pickle.dump(model,f)
    f.close()

def loadmodel(namename):
    global model
    f = open('%s.obj'%namename,'rb')
    model = pickle.load(f)
    f.close()

def genmodel(bobo,device):
    global model, H, modeltype
    # Use the nn package to define our model as a sequence of layers. nn.Sequential
    # is a Module which contains other Modules, and applies them in sequence to
    # produce its output. Each Linear Module computes output from input using a
    # linear function, and holds internal Tensors for its weight and bias.
    # After constructing the model we use the .to() method to move it to the
    # desired device.
    if modeltype == '1-hidden':
        print('using %d hidden units'%(H))
        model = torch.nn.Sequential(
                  torch.nn.Linear(bobo.D_in, H),
                  torch.nn.ReLU(),
                  torch.nn.Linear(H, bobo.D_out),
                ).to(device)
    elif modeltype == '0-hidden':
        print('using simply fully connected output layer')
        model = torch.nn.Sequential(
                  torch.nn.Linear(bobo.D_in, bobo.D_out),
                ).to(device) # do i need the Sequential?

def genloss():
    global loss_fn
    # The nn package also contains definitions of popular loss functions; in this
    # case we will use Mean Squared Error (MSE) as our loss function. Setting
    # reduction='sum' means that we are computing the *sum* of squared errors rather
    # than the mean; this is for consistency with the examples above where we
    # manually compute the loss, but in practice it is more common to use mean
    # squared error as a loss by setting reduction='elementwise_mean'.
    #loss_fn = torch.nn.MSELoss(reduction='sum')

    loss_fn = torch.nn.MSELoss(reduction='mean')

def train(bobo,device):
    global model,loss_fn
    memstuff()
    if not model:
        genmodel(bobo,device)
    if not loss_fn:
        genloss()

    memstuff()
    print('before loop')

    for t in range(num_epochs):
      # Forward pass: compute predicted y by passing x to the model. Module objects
      # override the __call__ operator so you can call them like functions. When
      # doing so you pass a Tensor of input data to the Module and it produces
      # a Tensor of output data.
      y_pred = model(bobo.x) #l.61
      ###i can do a torch.max
      _, pred = torch.max(y_pred,1)
      _, real = torch.max(bobo.y,1)
      acc_ = (real == pred)
      myacc = torch.sum(acc_).item()/bobo.batchsize
      print('accuracy:%f'%(myacc ))

      #memstuff()
      #print('in loop')
      # Compute and print loss. We pass Tensors containing the predicted and true
      # values of y, and the loss function returns a Tensor containing the loss.
      loss = loss_fn(y_pred, bobo.y)

      print(t, loss.item())

      # Zero the gradients before running the backward pass.
      model.zero_grad()

      # Backward pass: compute gradient of the loss with respect to all the learnable
      # parameters of the model. Internally, the parameters of each Module are stored
      # in Tensors with requires_grad=True, so this call will compute gradients for
      # all learnable parameters in the model.
      loss.backward()

      # Update the weights using gradient descent. Each parameter is a Tensor, so
      # we can access its data and gradients like we did before.
      with torch.no_grad():
        for param in model.parameters():
          param.data -= learning_rate * param.grad

    print(t, loss.item())
    memstuff()
    return model

def evaluate(model, bobo):
      y_pred = model(bobo.x) #l.61
      ###i can do a torch.max
      _, pred = torch.max(y_pred,1)
      _, real = torch.max(bobo.y,1)
      acc_ = (real == pred)
      myacc = torch.sum(acc_).item()/bobo.batchsize
      print('accuracy:%f'%(myacc ))
      for i, aclass in enumerate(bobo.classes):
            thisset = (real == i)
            thispred = (pred == i)
            thisacc = torch.sum(thispred & thisset).item()/torch.sum(thisset).item()
            print('clss %s acc is %f'%(aclass,thisacc))
