# # [AI선배_선택강의유사도테이블]Calculating Similar Course with Glove
# #### Developer : Jinsook Lee

# #### Version update ########
# 0.0.1 : June 22nd 2021


############################################################################################################################
#
#06-22-2021 : utils.py 
#
############################################################################################################################

import pandas as pd
import numpy as np
from konlpy.tag import Mecab
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


def execute(query):
    pc.execute(query)
    return pc.fetchall()

def data_load(filename, cursor):
    product = cursor
    f = open(filename, 'r')
    
    text = ''
    while True:
        line = f.readline()        
        if not line: break
        a = str(line)
        text = text + a
    f.close()
    
    data = pd.read_sql(text, product) 
    print("#------Read SQL Completed!------#")
    return data

def map_cour_cd(data, cursor):
    mapping = data_load("./sql/cour_cd_map.txt", cursor)
    mapping_ = mapping.set_index('old_cour_cd')['recent_cour_cd'].to_dict()
    for old, new in tqdm(mapping_.items()):
        data['cour_cd'] = data['cour_cd'].str.replace(old, new)
    print("#------Mapping Completed!------#")
    return data

def prep_groupby(cursor):
    major_text = data_load("./sql/sim_course_goal_sql.txt",cursor)
    print("#------Data Loaded!------#")
    
    major_text = major_text.rename(columns = {'lec311_cour_cd':'cour_cd'})
    major_text = map_cour_cd(major_text, cursor)
    major_txt = major_text[['cour_cd','cour_nm','outline','goal']].drop_duplicates().astype(str)
    major_txt['text'] = major_txt['cour_nm']+" "+ major_txt['outline'] +" "+major_txt['goal']
    major_txt['text'] = major_txt['text'].str.replace("None", "")
    major_txt = major_txt[['cour_cd','text']].drop_duplicates().groupby(['cour_cd'])['text'].apply(lambda x: "".join(x)).reset_index()
    return major_txt

def tokenizer(text, pos=['NNG', 'NNP', 'VV', 'VA']):
    """
    text : 형태소 분석할 raw data
    return : 불용어 제거된 형태소
    """
    stopword = pd.read_csv("/root/jupyter_src/LJS/LJS_210121_elec_rec/20212R/Major_recom/stopwords.txt", encoding ='utf8')
    stopwords = stopword.transpose().iloc[0].tolist()
    BAD_SYMBOLS_RE = re.compile('[^ ㄱ-ㅣ가-힣#]')
    text = BAD_SYMBOLS_RE.sub('', text)
    m = Mecab()
    return [
         word for word, tag in m.pos(text)
        if len(word) > 1 and tag in pos and word not in stopwords]

def tk_drop(data):
    """
    delete rows after tokenizing
    """
    data = data.dropna().reset_index(drop=True)
    data['text'] = data['text'].str.strip()
    
    print("#------Tokeninzing...------#")
    data['text_re'] = data['text'].apply(tokenizer)
    data = data.reset_index(drop=True)
    drop_idx = []

    for i in tqdm(range(len(data))):
        if len(data['text_re'].iloc[i])<1:
            print("drop_idx:", i, data['text_re'].iloc[i])
            drop_idx.append(int(i))
    data = data.drop(drop_idx)
    return data

def final_data():
    """
    Generate input data for Glove
    """
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
    product = pg.connect(product_connection_string)
        
    print("#------Data Preprocessing Start!------#")
    major_txt = prep_groupby(product)
    print("#------First prep completed!------#")
    data = tk_drop(major_txt)
    print("#------Second prep completed!------#")
    return data