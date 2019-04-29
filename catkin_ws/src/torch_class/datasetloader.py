#!/usr/bin/env python3
# -*- coding: utf-8 -*-
try:
    import cPickle as pickle
except:
    import pickle

import time

now = time.time()

def toc():
    print('toc. time: %f'%(time.time()-now))

pickle_in = open("a.obj",'rb')
labels = pickle.load(pickle_in)
toc()
print(type(labels))
xlist = pickle.load(pickle_in)
toc()
print(type(xlist))
