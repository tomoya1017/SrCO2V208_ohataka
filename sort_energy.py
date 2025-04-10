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
import re


N=128
chi = 10
ex_bx = 0.175
# JSON ファイルのパスを取得
# filtered_files = glob.glob("ene_with_state_n{}_with_128ini_{}bond_h{}_k*.json".format(N,chi,ex_bx))
# 正規表現パターン（k が偶数 & `_mpi4.json` 付き/なし）
# pattern = r"ene_with_state_n{}_with_128ini_{}bond_h{}_k(\d+)_mpi4.json".format(N, chi, ex_bx)

# ファイルリスト取得
filtered_files = glob.glob("ene_with_state_n{}_with_128ini_{}bond_h{}_k*_mpi4.json".format(N, chi, ex_bx))

# 条件に合うデータを格納
# filtered_files = []

# JSON ファイルをフィルタリング
# for file in json_files:
#     match = re.match(pattern, file)
#     if match:
#         k = int(match.group(1))  # k の数値部分を取得
#         if k in [0, 2, 4, 6,8,10,12,14]:  # k が 0, 2, 4, 6 の場合のみ対象
#             filtered_files.append(file)

# 結果を出力
all_data = []
i=0
for file in filtered_files:
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
with open('Sr_sorted_combined_n{}_with_n128ini_{}bond_h{}_even.json'.format(N,chi,ex_bx), 'w') as f:
    json.dump(sorted_all_data, f, indent=1, ensure_ascii=False)