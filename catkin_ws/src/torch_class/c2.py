#!/usr/bin/env python
# -*- coding: utf-8 -*-
#based on https://github.com/jcjohnson/pytorch-examples/blob/master/nn/two_layer_net_nn.py
# with some changes...

import torch
import numpy as np
import gc
try:
    import cPickle as pickle
except:
    import pickle

import rospy
from std_msgs.msg import Float32

import dl

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
QUIET = True

class SharedParam():
    def __init__(self,name, value):
        self.value = value
        self.name = name
        self.changed = False
        rospy.logdebug("instantiated SharedParam %s to value %s"%(self.name, self.value))
        rospy.set_param("~%s"%self.name,self.value)
    def set(self,value):
        self.value = value
        rospy.set_param("~%s"%self.name,self.value)
        self.changed = True
    def get(self):
        ###maybe catch here if value changed?
        oldvalue = self.value
        newvalue = rospy.get_param("~%s"%self.name,self.value)
        if not newvalue == oldvalue:
            self.value = newvalue
            self.changed = True
            rospy.logdebug("value from SharedParam %s changed to %s"%(self.name, self.value))

class CudaSet(dl.dataset):
    def __init__(self):
        super(CudaSet,self).__init__()
        self.device = torch.device("cuda:0") # Uncomment this to run on GPU
        self.model = None
        self.loss_fn = None
        self.H = 5000
        self.learning_rate = SharedParam('learning_rate', 0.000005)
        # self.learning_rate = 0.000005
        # rospy.set_param("~learning_rate",self.learning_rate)
        self.num_epochs = SharedParam('num_epochs', 10000)
        self.num_batches = SharedParam('num_batches', 100)
        # self.num_epochs = 10000
        self.modeltype = '1-hidden'
        self.genloss()
        ## needs to be done after stuff was sent to cuda!
        #self.gen()
        ### making it ROSSY, because, why not?
        self.losspub = rospy.Publisher('loss', Float32, queue_size=1)
        self.accpub = rospy.Publisher('accuracy', Float32, queue_size=1)
        #self.lrsub = rospy.Subscriber('lr',Float32, self.mycallback, queue_size=1)
        self.running = rospy.get_param('~running',0)
        self.dp = SharedParam('dp',0.8)
    # def mycallback(self,msg):
    #     self.learning_rate = msg.data
    #     rospy.loginfo('learning rate changed to %f'%self.learning_rate)

    def tocuda(self):
        dl.tic()
        #global x,y
        mytypexy = torch.tensor((),dtype=torch.float32, device=self.device)
        y = mytypexy.new_zeros((self.batchsize,self.numclasses))
        x = mytypexy.new_zeros((self.batchsize,self.xfeaturessize))
        for i,(item,thisTSNnnvalues) in enumerate(zip(self.labels,self.xlist)):
            x[i,:] = torch.from_numpy(np.asarray(thisTSNnnvalues)) ### this is ugly and probably very inefficient
            for j,classname in enumerate(self.classes):
                if item == classname:
                    y[i,j] = 1
        dl.toc()

        # N is batch size; D_in is input dimension;
        # H is hidden dimension; D_out is output dimension.
        #N, D_in, H, D_out = 64, 1000, 100, 10
        self.N, self.D_in, self.D_out = self.batchsize, self.xfeaturessize, self.numclasses

        self.x = x
        self.y = y
        if not QUIET:
            print(x.size())
            print(y.size())
        del x
        del y
        del mytypexy
        torch.cuda.empty_cache()


    def savemodel(self,namename):
        f = open('%s.obj'%namename,'wb')
        pickle.dump(self.model,f)
        f.close()

    def loadmodel(self,namename):
        f = open('%s.obj'%namename,'rb')
        self.model = pickle.load(f)
        f.close()

    def gen(self, modeltype):
        # Use the nn package to define our model as a sequence of layers. nn.Sequential
        # is a Module which contains other Modules, and applies them in sequence to
        # produce its output. Each Linear Module computes output from input using a
        # linear function, and holds internal Tensors for its weight and bias.
        # After constructing the model we use the .to() method to move it to the
        # desired device.
        self.modeltype = modeltype
        if self.modeltype == '1-hidden':
            print('using %d hidden units'%(self.H))
            self.model = torch.nn.Sequential(
                      torch.nn.Linear(self.D_in, self.H),
                      torch.nn.ReLU(),
                      torch.nn.Linear(self.H, self.D_out),
                    ).to(self.device)
        elif self.modeltype == '0-hidden':
            print('using simply fully connected output layer')
            self.model = torch.nn.Sequential(
                      torch.nn.Linear(self.D_in, self.D_out),
                    ).to(self.device) # do i need the Sequential?
        elif self.modeltype == '0-hiddenD':
            print('using fully connected output layer + dropout')
            # self.dp = 0.8
            self.model = torch.nn.Sequential(
                    torch.nn.Dropout(p=self.dp.value),
                    torch.nn.Linear(self.D_in, self.D_out)
                    ).to(self.device)
            self.std = 0.001
            self.mean = 0
            self.model.apply(self.init_weights)

        elif self.modeltype == '1-hiddenD':
            print('using fully connected output layer + dropout')
            # self.dp = 0.8
            self.model = torch.nn.Sequential(
                      torch.nn.Dropout(p=self.dp.value),
                      torch.nn.Linear(self.D_in, self.H),
                      torch.nn.ReLU(),
                      torch.nn.Linear(self.H, self.D_out)
                    ).to(self.device)
            self.std = 0.001
            self.mean = 0
            self.model.apply(self.init_weights)

        elif self.modeltype == '0-hidden2':
            print('using simply fully connected output layer')
            self.model = torch.nn.Sequential(
                      torch.nn.Linear(self.D_in, self.D_out),
                      torch.nn.ReLU(),
                    ).to(self.device) # do i need the Sequential?

    def change_dropoutparam(self,m):
        if type(m) == torch.nn.Dropout:
            print('actually running dropout changing bit!')
            m.p = self.dp.value
            # m = torch.nn.Dropout(p=self.dp.value)
    def check_dropoutparam(self,m):
        if type(m) == torch.nn.Dropout:
            if 'p' in dir(m):
                print(m.p)

    def init_weights(self,m):
        if type(m) == torch.nn.Linear:
            # torch.nn.init.xavier_uniform(m.weight)
            # m.bias.data.fill_(self.weights)
            torch.nn.init.normal_(m.weight,mean=self.mean,std=self.std)

    def genloss(self):
        # The nn package also contains definitions of popular loss functions; in this
        # case we will use Mean Squared Error (MSE) as our loss function. Setting
        # reduction='sum' means that we are computing the *sum* of squared errors rather
        # than the mean; this is for consistency with the examples above where we
        # manually compute the loss, but in practice it is more common to use mean
        # squared error as a loss by setting reduction='elementwise_mean'.
        #loss_fn = torch.nn.MSELoss(reduction='sum')

        self.loss_fn = torch.nn.MSELoss(reduction='mean')


    def fit(self):
        self.running = 1
        rospy.set_param("~running",self.running)
        if not QUIET:
            memstuff()
            print('Training!')

        myacc = None
        outloss = None
        print("training for %d epochs"%self.num_epochs.value)
        self.num_epochs.changed = False
        for t in range(self.num_epochs.value):
          self.running = rospy.get_param("~running")
          self.num_epochs.get()
          self.learning_rate.get()
          self.dp.get()
          if self.running: ### to make this interruptible using ros params. should probably have used a service call...
              # Forward pass: compute predicted y by passing x to the model. Module objects
              # override the __call__ operator so you can call them like functions. When
              # doing so you pass a Tensor of input data to the Module and it produces
              # a Tensor of output data.
              y_pred = self.model(self.x) #l.61
              ###i can do a torch.max
              _, pred = torch.max(y_pred,1)
              _, real = torch.max(self.y,1)
              acc_ = (real == pred)
              myacc = torch.sum(acc_).item()/self.batchsize
              if not QUIET:
                  print('accuracy:%f'%(myacc ))
              self.accpub.publish(myacc)

              #memstuff()
              #print('in loop')
              # Compute and print loss. We pass Tensors containing the predicted and true
              # values of y, and the loss function returns a Tensor containing the loss.
              loss = self.loss_fn(y_pred, self.y)
              if not QUIET:
                  print(t, loss.item())
              self.losspub.publish(loss.item())

              # Zero the gradients before running the backward pass.
              self.model.zero_grad()

              # Backward pass: compute gradient of the loss with respect to all the learnable
              # parameters of the model. Internally, the parameters of each Module are stored
              # in Tensors with requires_grad=True, so this call will compute gradients for
              # all learnable parameters in the model.
              loss.backward()

              ## I will make the learning_rate variable to test it faster
              # new_learning_rate = rospy.get_param("~learning_rate")
              # if not new_learning_rate == self.learning_rate:
              if self.learning_rate.changed:
                  print('changing learning_rate to %f'%self.learning_rate.value)
                  self.learning_rate.changed = False
                  # self.learning_rate = new_learning_rate
              # new_dp = rospy.get_param("")
              if self.dp.changed:
                  ##update dropout layer!
                  self.model.apply(self.change_dropoutparam)
                  print('changing dropout_rate to %f'%self.dp.value)
                  self.dp.changed = False
                  self.model.apply(self.check_dropoutparam)

              # Update the weights using gradient descent. Each parameter is a Tensor, so
              # we can access its data and gradients like we did before.
              with torch.no_grad():
                for param in self.model.parameters():
                  param.data -= self.learning_rate.value * param.grad

              outloss = loss.item()
        return myacc, outloss#loss.item()
        print('Accuracy on training set: %f'%(myacc ))
        print(t, loss.item())
        #memstuff()
        #return model

    def evaluate(self, bobo):
          y_pred = self.model(bobo.x) #l.61
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

    def score(self, x,t ):
         y_pred = self.model(x) #l.61
         ###i can do a torch.max
         _, pred = torch.max(y_pred,1)
         _, real = torch.max(t,1)
         acc_ = (real == pred)
         myacc = torch.sum(acc_).item()/len(t)
         #print('accuracy:%f'%(myacc ))
         return myacc

    def predict(self, x):
         y_pred = self.model(x) #l.61
         _, pred = torch.max(y_pred,1)
         return [self.classes[i] for i in pred]

def memstuff():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj),obj.size())
