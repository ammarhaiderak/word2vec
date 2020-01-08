#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 02:43:36 2019

@author: ammar
"""
import gensim
from gensim.test.utils import datapath
from gensim import utils

class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""

    def __iter__(self):
        corpus_path = datapath('/run/media/ammar/Backup/Question1.cor')
        for line in open(corpus_path,encoding='utf-8',errors='ignore'):
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(line)
            


sentenses=MyCorpus()

#print(sentenses)

model=gensim.models.Word2Vec(sentenses)


for i, word in enumerate(model.wv.vocab):
    if i == 10:
        break
    print(word)
    
'''
Clean 
Unclean 
Amazed 
friendly    
''' 
    
    
while(1):    
    wordtosearch=input('Enter word to lookup: ')
    wordtosearch=wordtosearch.lower()    
    print(model.wv.most_similar(wordtosearch))


