import math
import context
import time
import torch
import numpy as np
import scipy as scipy
import scipy.linalg as linalg
import scipy.sparse.linalg as spr_linalg
import argparse
import config as cfg
from mps.mps import *
# from models import heisenbergs1
from complex_num.complex_operation import *
from optim.ad_optim import optimize_state
#from optim.ad_optim_lbfgs_mod import optimize_state
import unittest
import logging
log = logging.getLogger(__name__)
import json
import glob

N=8
chi = 20
ex_bx = 0
# JSON ファイルのパスを取得
json_files = glob.glob('ene_with_state_{}bond__{}site_h{}_k*.json'.format(chi,N,ex_bx))

# 全ての JSON ファイルを読み込み、辞書に結合
all_data = []
i=0
for file in json_files:
    print(file)
    i+=1
    with open(file, 'r') as f:
        data = json.load(f)
        # for i in data["ene"]:
        #     data["ene"] = i
        all_data.extend(data)
print(i)
# 辞書のキーでソートして新しい辞書を作成
sorted_all_data = sorted(all_data,key=lambda x: x["ene"])
print(len(all_data))
 
# 結合されたデータを新しい JSON ファイルに保存
with open('sorted_combined_{}bond_{}site_h{}.json'.format(chi,N,ex_bx), 'w') as f:
    json.dump(sorted_all_data, f, indent=1, ensure_ascii=False)