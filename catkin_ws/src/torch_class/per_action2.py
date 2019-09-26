#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sklearn.ensemble
from sklearn.metrics import confusion_matrix

import traceback

import rospy
from std_msgs.msg import Float32

'''loading script for the combined samples from both networks'''

import dl
import al
import c2
import pickle

class Options():
    def __init__(self):
        self.whichtotrain = None
        self.whichtotest = None
        self.num_epochs = 1000
        self.num_batches = 100
        self.modeltype = '0-hidden'
        self.learning_rate = 0.00005
        self.fr_val = None
        self.split = 0
        self.NUM = 0
        self.evaluation_mode = 'instant'
        self.choose_list = []
        self.mytester = None
    def settrain(self,which):
        self.whichtotrain = self._set(which)
        self.train = which
    def settest(self,which):
        self.whichtotest = self._set(which)
        self.test = which
    def _set(self,which):
        if which == 'hmdb14':
            return [0]
        elif which == 'both':
            return [0,1]
        elif which == 'myset':
            return [1]
    def __str__(self):
        me = ''
        for prop in dir(self):
            if not prop[0] == '_':
                propval = eval('self.%s.__str__()'%prop)
                me = '%s %s:%s\n'%(me,prop, propval )
        return me

##thanks SO https://stackoverflow.com/questions/1518522/find-the-most-common-element-in-a-list
def most_common(lst):
    return max(set(lst), key=lst.count)

class FSOut():
    def __init__(self):
        pass

def fsw(predfun,x,y,m,em, CF=True):
    #import pdb; pdb.set_trace()

    plist = predfun(x)
    if em == "both":
        testacc1, a1, thisout1 = fancyscores_cf( x,y, m, plist, 'instant', CF=CF)
        testacc2, a2, thisout2 = fancyscores_cf( x, y, m, plist, 'all', CF=CF)

        testacc = [testacc1,testacc2]
        a = [a1,a2]
        out = [thisout1, thisout2]
    else:
        testacc, a, out = fancyscores_cf( x,ys, m, plist, em, CF=CF)

    return testacc, a, out

def fancyscores_cf(x, y, m, plist, evaluation_mode, CF=True):
    cf_matrix = None
    thisout = FSOut()
    if evaluation_mode == 'instant':
        if CF:
            cf_matrix = confusion_matrix(y, plist)
        ##I need to do this by myself!
        bobo = [yi == yhi for yi,yhi in zip(y, plist) ]
        #acc = scorefun(x,y)
        acc = sum(bobo)/len(bobo)

    elif evaluation_mode == 'all':
        assert m ##it should not be empty
        movie = 0
        realy = []
        realyhat = []
        thispredstack = []
        for i,(yi,mi,yhi) in enumerate(zip(y,m,plist)):
            assert len(realy) == len(realyhat)
            if mi == movie:
                thispredstack.append(yhi)
                thisy = yi
            if mi > movie:
                ##the movie in thispredstack is complete!
                #calculate the mode of the prediction stack
                yh = most_common(thispredstack)
                realy.append(thisy)
                realyhat.append(yh)
                ## restart thispredstack
                thispredstack = [yhi]
                thisy = yi
                movie = movie +1
        if CF:
            cf_matrix = confusion_matrix(realy, realyhat)
        bobo = [yi == yhi for yi,yhi in zip(realy,realyhat) ]
        acc = sum(bobo)/len(bobo)

        thisout.realy = realy
        thisout.realyhat = realyhat
    thisout.bobo = bobo
    thisout.acc = acc
    thisout.cf_matrix = cf_matrix

    print('############')
    print("accuracy %s: %f"%(evaluation_mode,acc))
    # print("accuracy test: %f"%clf.score(testset.xlist, testset.labels))
    print('############')

    return acc, cf_matrix, thisout

def getcf2(trainsetstr,testsetstr, options):
    clf = sklearn.ensemble.RandomForestClassifier(n_estimators=100, random_state=0)#, max_features=1)
    trainset = dl.dataset()
    testset = dl.dataset()
    print('loading set')
    trainset.load([trainsetstr[i] for i in options.whichtotrain])
    print('limiting classes')
    trainset.choose_classes(options.choose_list)
    print('training, this may take a while')
    clf.fit(trainset.xlist, trainset.labels)
    testset.load([testsetstr[i] for i in options.whichtotest])
    print('limiting classes')
    testset.choose_classes(options.choose_list)
    print('calculating scores')

    testacc, a, out = fsw(clf.predict, testset.xlist, testset.labels, testset.movienum, options.evaluation_mode)

    return a, options, out

class Plotter():
    def __init__(self):
        rospy.init_node('plotter')
        self.testaccpub = rospy.Publisher('testacc', Float32, queue_size=1)
        self.testaccallpub = rospy.Publisher('testaccall', Float32, queue_size=1)
        self.trainaccpub = rospy.Publisher('trainacc', Float32, queue_size=1)
        self.losspub = rospy.Publisher('lossb', Float32, queue_size=1)
        self.splitpub = rospy.Publisher('split', Float32, queue_size=1)
        self.fr_pub = rospy.Publisher('f_or_r', Float32, queue_size=1)
        self.options_pub = rospy.Publisher('options', Float32, queue_size=1)



def getcf(trainsetstr,testsetstr, options):
    myplotter = Plotter()
    for set1,set2 in zip(trainsetstr,testsetstr):
        if 'rgb' in set1 and 'rgb' in set2:
            #fr_pub.publish(0.5)
            options.fr_val = 1
        elif 'flow' in set1 and 'flow' in set2:
            #fr_pub.publish(0.99)
            options.fr_val = 2
        else:
            #fr_pub.publish(0.1)
            options.fr_val = -1
    trainset = c2.CudaSet()
    testset = c2.CudaSet()
    trainonthis = [trainsetstr[i] for i in options.whichtotrain]
    trainset.load(trainonthis)
    trainset.choose_classes(options.choose_list)
    trainset.tocuda()
    trainset.gen(options.modeltype)
    trainset.num_epochs.set(options.num_epochs)
    trainset.learning_rate.set(options.learning_rate)
    trainset.num_batches.set(options.num_batches)
    testonthis = [testsetstr[i] for i in options.whichtotest]
    testset.load(testonthis)
    testset.choose_classes(options.choose_list)
    testset.tocuda()
    print(options)
    print('Training')
    i = 0
    while(i<trainset.num_batches.value):
        i +=1
    #for i in range(options.num_batches): ## i need this to be changeable in the middle of the loop!
        trainacc, loss = trainset.fit()
        ## here there will be changes depending on the evaluation mode
        testacc, _, __ = fsw(trainset.predict, testset.x, testset.labels, testset.movienum, options.evaluation_mode, CF=False)
        # testacc = trainset.score(testset.x, testset.y)
        myplotter.testaccpub.publish(testacc[0])
        myplotter.testaccallpub.publish(testacc[1])
        myplotter.trainaccpub.publish(trainacc)
        myplotter.losspub.publish(loss)
        myplotter.options_pub.publish(options.NUM)
        myplotter.splitpub.publish(options.split)
        myplotter.fr_pub.publish(options.fr_val)

    testacc, a, thisout = fsw(trainset.predict, testset.x, testset.labels, testset.movienum, options.evaluation_mode)
    # print('############')
    # print("accuracy test: %f"%testacc)
    # # print("accuracy test: %f"%trainset.score(testset.x, testset.y))
    # print('############')

    # plist = trainset.predict(testset.x)
    # a = confusion_matrix(testset.labels, plist)

    return a, options, thisout

#print(__name__)

if 0:# __name__ == "__main__":
    try:
        rospy.init_node('plotter')
        testaccpub = rospy.Publisher('testacc', Float32, queue_size=1)
        trainaccpub = rospy.Publisher('trainacc', Float32, queue_size=1)
        losspub = rospy.Publisher('lossb', Float32, queue_size=1)
        splitpub = rospy.Publisher('split', Float32, queue_size=1)
        fr_pub = rospy.Publisher('f_or_r', Float32, queue_size=1)
        options_pub = rospy.Publisher('options', Float32, queue_size=1)

        myoptions = []
        thisop = Options()
        thisop.settrain('hmdb14')
        thisop.settest('hmdb14')
        thisop.choose_list = ['brush_hair','chew','clap','drink','eat','jump','pick','pour','sit','smile','stand','talk','walk','wave']
        thisop.evaluation_mode = "all"
        myoptions.append(thisop)
        #
        # thisop = Options()
        # thisop.settrain('hmdb14')
        # thisop.settest('myset')
        # myoptions.append(thisop)
        #
        # thisop = Options()
        # thisop.settrain('both')
        # thisop.settest('myset')
        # myoptions.append(thisop)
        #
        # thisop = Options()
        # thisop.settrain('myset')
        # thisop.settest('myset')
        # myoptions.append(thisop)

        results = []


    # hmdb51testscores_own_flow_split_1_y_yhat1559680138.96.obj
    # hmdb51testscores_own_flow_split_2_y_yhat1559681846.25.obj
    # hmdb51testscores_own_flow_split_3_y_yhat1559683493.16.obj
    # hmdb51testscores_own_rgb_split_1_y_yhat1559680057.25.obj
    # hmdb51testscores_own_rgb_split_2_y_yhat1559681761.51.obj
    # hmdb51testscores_own_rgb_split_3_y_yhat1559683414.2.obj
    # hmdb51trainscores_own_flow_split_1_y_yhat1558476156.41.obj
    # hmdb51trainscores_own_flow_split_2_y_yhat1558489246.06.obj
    # hmdb51trainscores_own_flow_split_3_y_yhat1558502294.88.obj
    # hmdb51trainscores_own_rgb_split_1_y_yhat1558475489.24.obj
    # hmdb51trainscores_own_rgb_split_2_y_yhat1558488594.63.obj
    # hmdb51trainscores_own_rgb_split_3_y_yhat1558501624.87.obj


        for z,thisOptions in enumerate(myoptions):
            #options_pub.publish(z/float(len(myoptions)))
            thisOptions.NUM = z
            for i in range(1,4):
                #splitpub.publish(i)
                thisOptions.split = i
                if i == 1:    ## split 1 train
                    ## hmdb51, myset/ actually hmdb51 is hmdb14!
                    #hmdb train
                    #myset train
                    #hmdb test
                    #myset test
                    a,thisOptions = getcf2([
                    '/mnt/share/ar/datasets/df_per_action/hmdb51trainscores_own_rgb_split_1_y_yhat1558475489.24.obj',
                    '/mnt/share/ar/datasets/df_per_action/it shouldnt try to load thisrgb!'],[
                    '/mnt/share/ar/datasets/df_per_action/hmdb51testscores_own_rgb_split_1_y_yhat1559680057.25.obj',
                    '/mnt/share/ar/datasets/df_per_action/global_pool_test_rgbsplit_1_y_yhat1557369944.32.obj'], thisOptions)
                    break

                    b,thisOptions = getcf([
                    '/mnt/share/ar/datasets/df_per_action/hmdb51trainscores_own_flow_split_1_y_yhat1558476156.41.obj',
                    '/mnt/share/ar/datasets/df_per_action/global_pool_train_flowsplit_1_y_yhat1557353106.0.obj',],[
                    '/mnt/share/ar/datasets/df_per_action/hmdb51testscores_own_flow_split_1_y_yhat1559680138.96.obj',
                    '/mnt/share/ar/datasets/df_per_action/global_pool_test_flowsplit_1_y_yhat1557369955.06.obj'], thisOptions)
                    totala = a
                    totalb = b
                if i == 2:# ## split 2 train
                    a,thisOptions = getcf([
                    '/mnt/share/ar/datasets/df_per_action/hmdb51trainscores_own_rgb_split_2_y_yhat1558488594.63.obj',
                    '/mnt/share/ar/datasets/df_per_action/global_pool_train_rgbsplit_2_y_yhat1557353417.19.obj'],[
                    '/mnt/share/ar/datasets/df_per_action/hmdb51testscores_own_rgb_split_2_y_yhat1559681761.51.obj',
                    '/mnt/share/ar/datasets/df_per_action/global_pool_test_rgbsplit_2_y_yhat1557370097.39.obj'], thisOptions)

                    b,thisOptions = getcf([
                    '/mnt/share/ar/datasets/df_per_action/hmdb51trainscores_own_flow_split_2_y_yhat1558489246.06.obj',
                    '/mnt/share/ar/datasets/df_per_action/global_pool_train_flowsplit_2_y_yhat1557353433.46.obj'],[
                    '/mnt/share/ar/datasets/df_per_action/hmdb51testscores_own_flow_split_2_y_yhat1559681846.25.obj',
                    '/mnt/share/ar/datasets/df_per_action/global_pool_test_flowsplit_2_y_yhat1557370107.53.obj'], thisOptions)
                    totala = a+ totala
                    totalb = b+ totalb

                if i == 3:# ## split 3 train
                    a,thisOptions = getcf([
                    '/mnt/share/ar/datasets/df_per_action/hmdb51trainscores_own_rgb_split_3_y_yhat1558501624.87.obj',
                    '/mnt/share/ar/datasets/df_per_action/global_pool_train_rgbsplit_3_y_yhat1557362421.94.obj'],[
                    '/mnt/share/ar/datasets/df_per_action/hmdb51testscores_own_rgb_split_3_y_yhat1559683414.2.obj',
                    '/mnt/share/ar/datasets/df_per_action/global_pool_test_rgbsplit_3_y_yhat1557370257.33.obj'], thisOptions)

                    b,thisOptions = getcf([
                    '/mnt/share/ar/datasets/df_per_action/hmdb51trainscores_own_flow_split_3_y_yhat1558502294.88.obj',
                    '/mnt/share/ar/datasets/df_per_action/global_pool_train_flowsplit_3_y_yhat1557362437.09.obj'],[
                    '/mnt/share/ar/datasets/df_per_action/hmdb51testscores_own_flow_split_3_y_yhat1559683493.16.obj',
                    '/mnt/share/ar/datasets/df_per_action/global_pool_test_flowsplit_3_y_yhat1557370264.5.obj'], thisOptions)
                    totala = a+ totala
                    totalb = b+ totalb
                print('############')
                results.append(a)
                results.append(b)


            print('############')
            results.append(totala)
            results.append(totalb)
            ####tests only one
            #break

        with open('results%s.obj'%dl.now,'wb') as f:
            pickle.dump(results,f)
            pickle.dump(myoptions,f)

    except BaseException as E:
        print(E)
    except:
        traceback.print_exc(file=sys.stdout)
