#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys


try:
    import cPickle as pickle
except:
    import pickle

import time

import traceback

from std_msgs.msg import Float32MultiArray


##we will try it with globals
#x = None
#y = None

QUIET = False

def tic():
    global now
    now = time.time()

def toc():
    print('toc. time: %f'%(time.time()-now))

class dataset():
    def __init__(self):
        self.labels = []
        self.xlist = []
        self.movienum = []
    def _loadload(self,filename):
        pickle_in = open(filename,'rb')
        if self.labels:
            print('labels not empty. extending set!')

        ### so I've changed the order in which I save xlist and label list. i suck.
        ##now I need to check this
        # tic()
        # xlist = pickle.load(pickle_in)
        # print(xlist[0])
        # toc()
        #
        # tic()
        # labelist = pickle.load(pickle_in)
        # print(labellist[0])
        # toc()
        alist = pickle.load(pickle_in)
        blist = pickle.load(pickle_in)
        if type(alist[0]) == type(''):
            ###then this is a label! so it should be like
            xlist = blist
            labelist = alist
        elif type(blist[0]) == type(''):
            ###then this is a label! so it should be like
            xlist = alist
            labelist = blist
        else:
            print('Error: could not make sense of pickle types!')

        if type(xlist[0]) == type([]):
            print('File seems to have movie chunks! Using fancy parser!')
            mymultiarray = Float32MultiArray()
            ii = 0
            if type(xlist[1][0]) == type(mymultiarray):
                islistofmultiarray = True
            else:
                islistofmultiarray = False
            for ixlist,ilabel in zip(xlist,labelist):
                ###I've made a mistake and the first list was empty. I will check for this and only add if the list is not empty:
                if ixlist:
                    if islistofmultiarray:
                        xtolist = [iixlist.data for iixlist in ixlist]
                    else:
                        xtolist = ixlist
                    # if not QUIET:
                    #     print(xtolist)
                    self.xlist.extend(xtolist)
                    ltolist = [ilabel for nope in range(len(ixlist))]
                    # if not QUIET:
                    #     print(ltolist)
                    self.labels.extend(ltolist)
                    mtolist = [ii for nope in range(len(ixlist))]
                    # if not QUIET:
                    #     print(mtolist)
                    self.movienum.extend(mtolist)
                    ii = ii + 1
                else:
                    print('I ve messed up. this list is empty!')
                    pass # don't add the label if the ixlist for the label was incorrectly formed
        else:
            print('File does not seem to have movie chunks! Using normal parser')
            self.xlist.extend(xlist)
            self.labels.extend(labelist )
        #print(type(self.xlist))



        #print(type(self.labels))


        pickle_in.close()

    def load(self,filenameL):
        print('loading a suposedly big database. this will take a while (~2 minutes for a 10gb set)')
        try: ### overloading python style(?)...
            if type(filenameL) == type([]):
                for filename in filenameL:
                    self._loadload(filename)

            elif  type(filenameL) == type(''):
                filename = filenameL
                self._loadload(filename)

            self._update_classes()

        except:
            traceback.print_exc(file=sys.stdout)

    def choose_classes(self, choose_list):
        if choose_list:
            newlist = []
            newscores = []
            for thisLabel,thisScore in zip(self.labels,self.xlist):
                if thisLabel in choose_list:
                    newlist.append(thisLabel)
                    newscores.append(thisScore)

            self.labels = newlist
            self.xlist = newscores

            self._update_classes()
        else:
            print('choose list is empty! using previous list with %d elements'%self.numclasses)

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
        if not QUIET:
            print("num of classes %d"%self.numclasses)
            print("classes are:")
            print(self.classes)
            print("batch size %d"%self.batchsize)
            print("feature size %d"%self.xfeaturessize)

            print('Done!')
