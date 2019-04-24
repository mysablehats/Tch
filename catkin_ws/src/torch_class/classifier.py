#!/usr/bin/env python
# -*- coding: utf-8 -*-
#copy pasted example from https://github.com/jcjohnson/pytorch-examples/blob/master/nn/two_layer_net_nn.py
import torch
try:
    import cPickle as pickle
except:
    import pickle
import numpy as np

import time

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

now = time.time()
device = torch.device("cuda:0") # Uncomment this to run on GPU
model = None

def toc():
    print('toc. time: %f'%(time.time()-now))

def train(datasetloader):
    #print(dir(datasetloader))
    global model
    ###now make them into matrices, erm, tensors, whatever...
    ### this is maybe already implemented in pytorch, I just don't know the name...
    classes = list(set(datasetloader.labels))
    classes.sort()
    numclasses = len(classes)
    batchsize = len(datasetloader.labels)
    xfeaturessize =  len(datasetloader.xlist[0])
    mytypexy = torch.tensor((),dtype=torch.float32, device=device)
    y= mytypexy.new_zeros((batchsize,numclasses))
    x = mytypexy.new_zeros((batchsize,xfeaturessize))
    for i,(item,thisTSNnnvalues) in enumerate(zip(datasetloader.labels,datasetloader.xlist)):
        x[i,:] = torch.from_numpy(np.asarray(thisTSNnnvalues)) ### this is ugly and probably very inefficient
        for j,classname in enumerate(classes):
            if item == classname:
                y[i,j] = 1


    #x = torch.stack(xlist)

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    #N, D_in, H, D_out = 64, 1000, 100, 10
    N, D_in, H, D_out = batchsize, xfeaturessize, 3000, numclasses

    # Create random Tensors to hold inputs and outputs
    #x = torch.randn(N, D_in, device=device)
    #y = torch.randn(N, D_out, device=device)

    print(x.size())
    print(y.size())
    toc()
    # Use the nn package to define our model as a sequence of layers. nn.Sequential
    # is a Module which contains other Modules, and applies them in sequence to
    # produce its output. Each Linear Module computes output from input using a
    # linear function, and holds internal Tensors for its weight and bias.
    # After constructing the model we use the .to() method to move it to the
    # desired device.
    model = torch.nn.Sequential(
              torch.nn.Linear(D_in, H),
              torch.nn.ReLU(),
              torch.nn.Linear(H, D_out),
            ).to(device)

    # The nn package also contains definitions of popular loss functions; in this
    # case we will use Mean Squared Error (MSE) as our loss function. Setting
    # reduction='sum' means that we are computing the *sum* of squared errors rather
    # than the mean; this is for consistency with the examples above where we
    # manually compute the loss, but in practice it is more common to use mean
    # squared error as a loss by setting reduction='elementwise_mean'.
    #loss_fn = torch.nn.MSELoss(reduction='sum')
    loss_fn = torch.nn.MSELoss(reduction='mean')

    learning_rate = 1e-3
    for t in range(500):
      # Forward pass: compute predicted y by passing x to the model. Module objects
      # override the __call__ operator so you can call them like functions. When
      # doing so you pass a Tensor of input data to the Module and it produces
      # a Tensor of output data.
      y_pred = model(x)

      # Compute and print loss. We pass Tensors containing the predicted and true
      # values of y, and the loss function returns a Tensor containing the loss.
      loss = loss_fn(y_pred, y)
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
    toc()
