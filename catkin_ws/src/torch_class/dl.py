#!/usr/bin/env python
# -*- coding: utf-8 -*-
#based on https://github.com/jcjohnson/pytorch-examples/blob/master/nn/two_layer_net_nn.py
# with some changes...

import sys

import torch
try:
    import cPickle as pickle
except:
    import pickle
import numpy as np

import time

import traceback


##we will try it with globals
#x = None
#y = None

device = torch.device("cuda:0") # Uncomment this to run on GPU

def tic():
    global now
    now = time.time()

def toc():
    print('toc. time: %f'%(time.time()-now))

class dataset():
    def __init__(self,filename):
        print('loading a suposedly big database. this will take a while (~2 minutes for a 10gb set)')
        try:
            tic()
            pickle_in = open(filename,'rb')
            self.labels = pickle.load(pickle_in)
            toc()
            tic()
            print(type(self.labels))
            self.xlist = pickle.load(pickle_in)
            toc()
            print(type(self.xlist))

            #print(dir(datasetloader))
            ###now make them into matrices, erm, tensors, whatever...
            ### this is maybe already implemented in pytorch, I just don't know the name...

            self._update_classes()

        except:
            traceback.print_exc(file=sys.stdout)

    def choose_classes(self, choose_list):
        newlist = []
        newscores = []
        for thisLabel,thisScore in zip(self.labels,self.xlist):
            if thisLabel in choose_list:
                newlist.append(thisLabel)
                newscores.append(thisScore)

        self.labels = newlist
        self.xlist = newscores

        self._update_classes()

    def save(self,filename):
        with open(filename,'wb') as f:
            pickle.dump(self.labels,f)
            pickle.dump(self.xlist,f)

    def _update_classes(self):
        classes = list(set(self.labels))
        classes.sort()

        self.batchsize  = len(self.labels)
        self.xfeaturessize =  len(self.xlist[0])
        self.classes = classes
        self.numclasses =  len(classes)
        print("num of classes %d"%self.numclasses)
        print("classes are:")
        print(self.classes)
        print("batch size %d"%self.batchsize)
        print("feature size %d"%self.xfeaturessize)

        print('Done!')

    def tocuda(self, numunits=1000):
        tic()
        #global x,y
        mytypexy = torch.tensor((),dtype=torch.float32, device=device)
        y = mytypexy.new_zeros((self.batchsize,self.numclasses))
        x = mytypexy.new_zeros((self.batchsize,self.xfeaturessize))
        for i,(item,thisTSNnnvalues) in enumerate(zip(self.labels,self.xlist)):
            x[i,:] = torch.from_numpy(np.asarray(thisTSNnnvalues)) ### this is ugly and probably very inefficient
            for j,classname in enumerate(self.classes):
                if item == classname:
                    y[i,j] = 1
        toc()

        # N is batch size; D_in is input dimension;
        # H is hidden dimension; D_out is output dimension.
        #N, D_in, H, D_out = 64, 1000, 100, 10
        self.N, self.D_in, self.H, self.D_out = self.batchsize, self.xfeaturessize, numunits, self.numclasses

        self.x = x
        self.y = y
        print(x.size())
        print(y.size())
        del x
        del y
        del mytypexy
        torch.cuda.empty_cache()
