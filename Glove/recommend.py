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

def connect():
    user = 
    password = 
    host_product = 
    dbname = 
    port = 

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
    return product
    

def prep():
    product = connect()
    filter_reg = data_load("./course_reg.txt", product) #course taken
    rgcn = data_load("./rgcn_elec_now_open.txt", product) #rgcn
    
    filter_reg = filter_reg[['std_id','cour_cd']].drop_duplicates()
    filter_reg['key'] = filter_reg['std_id'] + filter_reg['cour_cd']
    drp_list = filter_reg['key'].tolist()
    del filter_reg

    rgcn['key'] = rgcn['std_id']+rgcn['cour_cd']
    rgcn_f = rgcn[~rgcn['key'].isin(drp_list)][['std_id','cour_cd','cour_nm','score']]
    del rgcn
           
    return rgcn_f

class Recommend:

    def __init__(self):
        self.rgcn_f = prep() 
        
    #최초페이지용 함수
    def initial_load(self, std_id):
        first_rec = self.rgcn_f[self.rgcn_f['std_id']==std_id][['std_id','cour_cd','cour_nm','score']]
        return first_rec

    #추천받기 누른 후 함수
    def final_score(self, std_id, click_list):
        product = connect()
        first_rec = self.initial_load(std_id)
        
        q = f"""
      
        """
        glove_score = pd.read_sql(q, product)
        print("#------Here is ",std_id,"'s Recommended Courses!------#")
        naive_rec_list = first_rec['cour_cd'].tolist() #rgcn initial cour_cd list by std
        
        #similarity filtering between clicked courses and rgcn naive recommended courses
        glove_score = glove_score[(glove_score['cour_cd2'].isin(naive_rec_list))&(glove_score['cour_cd1'].isin(click_list))] 
        
        #rgcn sum
        glove_score = glove_score.groupby(['cour_cd2']).sum().reset_index().rename(columns = {'cour_cd2':'cour_cd'})
       
        first_rec = pd.merge(first_rec, glove_score, how= 'left').fillna(0)

        #arithetic mean of rgcn_score &  avg_glove_score
        first_rec['final_score'] = (first_rec['score'] + first_rec['similarity'])/(1+len(click_list))

        #harmonic mean of rgcn_score & avg_glove_score
        #first_rec['final_score'] = 2*(first_rec['score'] * first_rec['avg_score'])/(first_rec['score']+first_rec['avg_score'])
        return first_rec.sort_values(by ='final_score', ascending =False)       
    
    def course_rec(self, std_id, click_list):
        if len(click_list) == 0:
            rec_list = self.initial_load(std_id)
        else:
            rec_list = self.final_score(std_id, click_list)
        return print(rec_list)
