# -*- coding: utf-8 -*-
"""
Created on Tue May  7 09:58:40 2019

@author: ZHYQAQ
"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
def BLUinit():
    
#    dic = {'conv_1_1': {'ws': [3, 3, 3, 32], 'bs': [32]},
#           
#           'dense_1_1': {'ln': 7, 'gr': 32, 'ks': 3},
#           
#           'dense_1_2': {'ln': 5, 'gr': 24, 'ks': 5},
#           
#           'dense_2_1': {'ln': 3, 'gr': 16, 'ks': 7},
#           
#           'dense_1_3': {'ln': 3, 'gr': 16, 'ks': 7},
#           
#           'dense_1_4': {'ln': 5, 'gr': 24, 'ks': 5},
#           
#           'dense_1_5': {'ln': 7, 'gr': 32, 'ks': 3}}
#    
#    return dic
    
    dic = {# 'conv_1_1': {'ws': [3, 3, 3, 32], 'bs': [32]}, # for local and newtork
           'conv_1_1': {'ws': [3, 3, 3, 32], 'bs': [32]},   # for acdc
           #'conv_1_2': {'ws': [3, 1, 32, 32], 'bs': [32]},
           
           'dense_1_1': {'ln': 7, 'gr': 16, 'ks': 3},
           
           'dense_1_2': {'ln': 5, 'gr': 24, 'ks': 5},
           
           'dense_2_1': {'ln': 3, 'gr': 32, 'ks': 7},
           
           'dense_1_3': {'ln': 3, 'gr': 32, 'ks': 7},
           
           'dense_1_4': {'ln': 5, 'gr': 24, 'ks': 5},
           
           'dense_1_5': {'ln': 7, 'gr': 16, 'ks': 3}}
    
    return dic

def FCDdenseinit():
    
    dic = {'dense_1': {'laynum': 3, 'gr': 36, 'ks': 3},
           
           'dense_2': {'laynum': 5, 'gr': 36, 'ks': 5},
           
           'dense_3': {'laynum': 7, 'gr': 36, 'ks': 7},
           
           'dense_4': {'laynum': 5, 'gr': 36, 'ks': 5},
           
           'dense_5': {'laynum': 3, 'gr': 36, 'ks': 3}}
    
    return dic
    
def KeepProbInitializer():
    
    keepProb = 1.0
    
    return keepProb

def TrainBatchSize():
    
    Ubatchsize = 4
    
    FCDbatchsize = 4
    
    batchsize = 1
    
    return batchsize, Ubatchsize, FCDbatchsize

def TrainEpoch():
    
    Uepoch = 20
    
    FCDepoch = 20
    
    BLUepoch = 40
    
    return Uepoch, FCDepoch, BLUepoch

def TrainNum():
    
    num = 5500
    
    return num

def lrinit(itr, maxitr):
    
    rate = 1e-4
    
    pct = itr / maxitr
    
    if pct < 0.3:
        
        lerate = rate
    
    elif pct > 0.3 and pct <= 0.5:
        
        lerate = rate / 10 
        
    elif pct > 0.5 and pct <= 0.7:
        
        lerate = rate / 10
        
    elif pct > 0.7  and pct <= 0.8:
        
        lerate = rate / 10 
        
    elif pct > 0.8 and pct < 0.9:
        
        lerate = rate / 100
        
    else:
        
        lerate = rate / 100
    
    return lerate