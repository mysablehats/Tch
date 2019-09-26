#!/usr/bin/env python
# -*- coding: utf-8 -*-
import dl

'''aproximate loader: combine 2 dataset models of the same split; flow and such
 to make another one with larger feature size '''

def aproximate_loader(rgbset,flowset):
    lastr = ''
    lastf = ''
    rfset = dl.dataset()

    for counter in range(max(rgbset.batchsize,flowset.batchsize)): ##we are going to check in the end if there are no samples left, so this is okay. 
        ## okay, so if one of them changed to the next class already, the other
        ## one needs to dump the amount of samples necessary for it to get to
        ## the next class
        if rgbset.labels[0] == flowset.labels[0]:
            #same class just concatenate and pop both lists.
            lastf = rgbset.labels[0]
            lastr = rgbset.labels[0]
            rfset.labels.append(rgbset.labels[0])
            rgbset.labels.pop(0)
            flowset.labels.pop(0)
            thisrxlist = list(rgbset.xlist[0])
            thisrxlist.extend(flowset.xlist[0])
            rfset.xlist.append(tuple(thisrxlist))
            rgbset.xlist.pop(0)
            flowset.xlist.pop(0)

        else:
            print('# WARNING: these guys dont line up for current counter:%d'%counter)
            if rgbset.labels[0] == lastr:
                while rgbset.labels[0] == lastr:
                    rgbset.labels.pop(0)
                    rgbset.xlist.pop(0)
                    print('dumping rgbset sample')
            elif flowset.labels[0] == lastf:
                while flowset.labels[0] == lastf:
                    flowset.labels.pop(0)
                    flowset.xlist.pop(0)
                    print('dumping flowset sample')
            if len(rgbset.labels) == 0 :
                print('dumping last %d flowset samples'%len(flowset.labels))

                break
            if len(flowset.labels) == 0 :
                print('dumping last %d rgbset samples'%len(rgbset.labels))
                break

    rfset.choose_classes(flowset.classes)


    return rfset
