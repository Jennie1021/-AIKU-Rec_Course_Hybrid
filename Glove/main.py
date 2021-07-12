# # [AI선배_선택강의유사도테이블]Calculating Similar Course with Glove
# #### Developer : Jinsook Lee

# #### Version update ########
# 0.0.1 : June 22nd 2021


############################################################################################################################
#
#06-22-2021 : main.py 
#
############################################################################################################################

import sys
import argparse
import pandas as pd
import pickle
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import psycopg2 as pg
from tqdm import tqdm
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from utils import final_data
from glove import Corpus, Glove

#Train function 
def train(text):
    corpus = Corpus()
    corpus.fit(text, window=args.window_size)
    # model
    glove = Glove(no_components=args.vector_size, learning_rate=0.01)     # 0.05
    glove.fit(corpus.matrix, epochs=args.n_epochs, no_threads=4, verbose=True)    # Wall time: 8min 32s'
    glove.add_dictionary(corpus.dictionary)
    glove.save('glove_w{}_epoch{}_size{}.model'.format(args.window_size, args.n_epochs, args.vector_size))
    return glove

def sent2vec_glove(tokens, word_dict):
    '''
    embedding tokens
    '''
    
    word_table = word_dict #glove에서 학습시킨 word dict
    matrix = np.mean(np.array([word_table[t] for t in tokens if t in word_table]), axis=0) 
    print("#------Matrix Generated!------#")
    return matrix

#main
def main(args):
        
    prep = final_data()
    prep.to_csv("course_keywords_chunk.txt", sep='\t')
    
    data = prep['text_re'].reset_index()['text_re']
    
    print("#------Train Start!------#")
    glove_model=train(data)
    
    print("#------Calculating Start!------#")
    
    # word dict 생성
    print("#------Word dictionary generate!------#")
    word_dict = {}
    for word in  glove_model.dictionary.keys():
        word_dict[word] = glove_model.word_vectors[glove_model.dictionary[word]]
    print('[Success !] Lengh of word dict... : ', len(word_dict))

    # save word_dict
    with open('glove_word_dict_{}.pickle'.format(args.vector_size), 'wb') as f:
        pickle.dump(word_dict, f)
    print('[Success !] Save word dict!...')
    
    

    mat = np.zeros((len(data), args.vector_size))
    for i in range(len(data)):
        a = sent2vec_glove(data[i], word_dict)
        mat[i] = a
    
    b = pd.DataFrame(cosine_similarity(mat,mat), index = prep.cour_cd, columns =prep.cour_cd)
    b.index.name = 'cour_cd1'
    
    final_cour_sim = pd.DataFrame(b.stack()).reset_index().rename(columns = {'cour_cd':'cour_cd2',0:'similarity'})    
    final_cour_sim.to_csv("glove_cour_similarity_vs{}.txt".format(args.vector_size), sep='\t')
    print("#------Result Saved------#")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--vector_size", type=int, default=200)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--n_epochs", type=int, default=200)
    
    args = parser.parse_args()
    print(args)
    
    main(args)

