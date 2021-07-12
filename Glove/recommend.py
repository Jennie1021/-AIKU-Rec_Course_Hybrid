import pandas as pd
import numpy as np
import argparse
#from scipy.stats.mstats import gmean, hmean
import itertools
import re
import os
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

from datetime import datetime
from tqdm import tqdm
import ast

import psycopg2 as pg
import cx_Oracle

import pickle
import gzip

from utils import data_load, map_cour_cd
from glove import Corpus, Glove
from sklearn.metrics.pairwise import cosine_similarity

def execute(query):
        pc.execute(query)
        return pc.fetchall()

def read_rgcn(filename):
    user = 'datahub'
    password = 'datahub123!@#'
    host_product = '163.152.11.12'
    dbname = 'pkuhub'
    port = '5432'

    product_connection_string = "dbname={dbname} user={user} host={host} password={password} port={port}"\
                                .format(dbname=dbname,
                                        user=user,
                                        host=host_product,
                                        password=password,
                                        port=port)
    try:
        product = pg.connect(product_connection_string)
    except:
        print('*****ERROR******')

        pc = product.cursor()

    rgcn = data_load("/root/jupyter_src/LJS/LJS_210121_elec_rec/20212R/Course Recom/RGCN/sql/course_reg.txt", product)

def sent2vec_glove(tokens, word_dict):
    '''
    embedding tokens
    '''

    word_table = word_dict #glove에서 학습시킨 word dict
    matrix = np.mean(np.array([word_table[t] for t in tokens if t in word_table]), axis=0) 
    print("#------Matrix Generated!------#")
    return matrix



def prep():
    user = 'datahub'
    password = 'datahub123!@#'
    host_product = '163.152.11.12'
    dbname = 'pkuhub'
    port = '5432'

    product_connection_string = "dbname={dbname} user={user} host={host} password={password} port={port}"\
                                .format(dbname=dbname,
                                        user=user,
                                        host=host_product,
                                        password=password,
                                        port=port)
    try:
        product = pg.connect(product_connection_string)
    except:
        print('*****ERROR******')

        pc = product.cursor()

    filter_reg = data_load("/root/jupyter_src/LJS/LJS_210121_elec_rec/20212R/Course Recom/RGCN/sql/course_reg.txt", product)
    rgcn = data_load("/root/jupyter_src/LJS/LJS_210121_elec_rec/20212R/Course Recom/Glove/sql/rgcn_elec_now_open.txt", product)

    filter_reg = filter_reg[['std_id','cour_cd']].drop_duplicates()
    filter_reg['key'] = filter_reg['std_id'] + filter_reg['cour_cd']
    drp_list = filter_reg['key'].tolist()
    del filter_reg

    rgcn['key'] = rgcn['std_id']+rgcn['cour_cd']
    rgcn_f = rgcn[~rgcn['key'].isin(drp_list)][['std_id','cour_cd','cour_nm','score']]
    del rgcn

    # load word_dict
    with open('glove_word_dict_300.pickle', 'rb') as f:
        word_dict = pickle.load(f)

    course_chunk = pd.read_csv("course_keywords_chunk.txt", sep = '\t')
    for i in range(len(course_chunk['text_re'])):
        course_chunk['text_re'][i] = ast.literal_eval(course_chunk['text_re'][i])

    data = course_chunk['text_re']               
    cour_list = course_chunk['cour_cd'].tolist()

    mat = np.zeros((len(data), 300))
    for i in range(len(data)):
        a = sent2vec_glove(data[i], word_dict)
        mat[i] = a
    return rgcn_f, mat, data, cour_list

class Recommend:
    def __init__(self):
        self.rgcn_f, self.mat, self.data, self.cour_list = prep() 

    #최초페이지용 함수
    def initial_load(self, std_id):
        first_rec = self.rgcn_f[self.rgcn_f['std_id']==std_id][['std_id','cour_cd','cour_nm','score']]
        return first_rec

    #추천받기 누른 후 함수
    def cal(self, std_id, cour_cd_chosen):
        """
        cour_cd_chosen : list of clicked courses
        """
        first_rec = self.initial_load(std_id)
        rgcn_list = first_rec.cour_cd.tolist()
        sim = []
        for i in rgcn_list:
            if i in self.cour_list:
                simm = cosine_similarity(self.mat,self.mat)[self.cour_list.index(i),self.cour_list.index(cour_cd_chosen)]
                print('Cosine similarity between',i,'and', cour_cd_chosen,':',simm)
                sim.append([i,simm])
        return sim

    def final_score(self, std_id, click_list):
        click_list = click_list
        first_rec = self.initial_load(std_id)

        score_ = []
        for i in click_list:
            if i in self.cour_list:
                similarity = self.cal(std_id, i)
                score_.append([i,similarity])
        score_df = pd.DataFrame(columns = ['cour_cd','avg_score'])

        for i in range(len(score_)):
            score_d = pd.DataFrame(score_[i][1], columns = ['cour_cd','avg_score'])
            score_df = score_df.append(score_d, ignore_index =True)

        score_df = score_df.groupby(['cour_cd']).sum(['score']).reset_index()

        first_rec = pd.merge(first_rec, score_df, how= 'left').fillna(0)

        #arithetic mean of rgcn_score &  avg_glove_score
        first_rec['final_score'] = (first_rec['score'] + first_rec['avg_score'])/(1+len(score_))

        #harmonic mean of rgcn_score & avg_glove_score
        #first_rec['final_score'] = 2*(first_rec['score'] * first_rec['avg_score'])/(first_rec['score']+first_rec['avg_score'])
        return first_rec.sort_values(by ='final_score', ascending =False)       

    def course_rec(self, std_id, click_list):
        if len(click_list) == 0:
            rec_list = self.initial_load(std_id)
        else:
            rec_list = self.final_score(std_id, click_list)
        return print(rec_list)