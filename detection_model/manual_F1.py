# tensorboard --logdir=lightning_logs
#MODEL_NAME = '/home0/y2020/s2010320/Sarcasm_detection/MHA-BiLSTM/model_transformers_stance_manual/data/student/s2010320/Sarcasm_detection/MHA-BiLSTM/model/0.7739130258560181' # 最初のbest(一応取っといてる)
import random
import glob
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertForSequenceClassification

#tensorflowは pytorch_lightningより先にimportしないとSegmentation faultになる
import tensorflow as tf
import pytorch_lightning as pl
#import tensorflow as tf

import csv
import pandas as pd
import numpy as np
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


#use_cuda = torch.cuda.is_available() and True
#device = torch.device("cuda" if use_cuda else "cpu")
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
#torch.device("cuda")

# 日本語の事前学習モデル
MODEL_NAME = '/Users/satogo/Library/CloudStorage/GoogleDrive-gogogou0110@gmail.com/マイドライブ/卒業研究　コード(AIX  バックアップ)/MHA-BiLSTM (12 16)/model_transformers_stance_manual/0.8294481847503103' #best (全データ F1 : 0.8987)
#MODEL_NAME = '/home0/y2020/s2010320/Sarcasm_detection/MHA-BiLSTM/model_transformers_stance_manual/0.8109551396316101'     #try


tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
bert_sc = BertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
)
#bert_sc = bert_sc.cuda()

# 各データの形式を整える
max_length = 128
dataset_for_loader = []

text_list =[]
label_list = []
pattern_matching_scores_list = []

num_t1_p = 0
num_f1_p = 0
num_t0_p = 0
num_f0_p = 0

df = pd.read_csv("/Users/satogo/Library/CloudStorage/GoogleDrive-gogogou0110@gmail.com/マイドライブ/卒業研究　コード(AIX  バックアップ)/MHA-BiLSTM (12 16)/pattern_matching_score(bert_sentence)_last.csv", sep=',')
for line in df.values:
    review_text = line[0]
    label = 1#3 - line[13]
    text_list.append(review_text)
    label_list.append(label)
    correct_label = line[8]

    #パターンマッチングスコアと正解ラベルを比較することでパターンマッチングスコアの精度を検証
    p_score = line[6]
    if  ((correct_label == 1) & ( p_score > 0) ):
        num_t1_p += 1
    elif  ((correct_label == 0) & ( p_score > 0) ):
        num_f1_p += 1
    elif  ((correct_label == 0) & ( p_score == 0) ):
        num_t0_p += 1
    elif  ((correct_label == 1) & ( p_score == 0) ):
        num_f0_p += 1

encoding = tokenizer(
    text_list,
    max_length=max_length,
    padding='max_length',
    truncation=True
)
encoding = { k: torch.tensor(v) for k, v in encoding.items() }
labels = torch.tensor(label_list).cuda()
accuracy_p = (num_t1_p + num_t0_p)/1150 #パターンマッチングスコアのaccuracy(1150は総数)
precision_p = num_t1_p/(num_t1_p + num_f1_p)
recall_p = num_t1_p/(num_t1_p + num_f0_p)
F1_score_p = (2*precision_p*recall_p)/(precision_p + recall_p) 



# 推論
with torch.no_grad():
    output = bert_sc.forward(**encoding)
scores = output.logits # 分類スコア
labels_predicted = scores.argmax(-1) # スコアが最も高いラベル
num_correct = (labels_predicted==labels).sum().item() # 正解数
accuracy = num_correct/labels.size(0) # 精度

num_t1 = ((labels_predicted == 1) & (labels_predicted == labels) ).sum().item()
num_f1 = ((labels_predicted == 1) & (labels == 0) ).sum().item()
num_t0 = ((labels_predicted == 0) & (labels_predicted == labels) ).sum().item()
num_f0 = ((labels_predicted == 0) & (labels == 1) ).sum().item()
precision = num_t1/(num_t1 + num_f1)
recall = num_t1/(num_t1 + num_f0)
F1_score = (2*precision*recall)/(precision + recall) 



#print("# scores:")
#print(scores.size())
#print("# predicted labels:")
#print(labels_predicted)
print("# t1 :", num_t1)
print("# f1 :", num_f1)
print("# t0 :", num_t0)
print("# f0 :", num_f0)
print("----- F1_score :", F1_score, "-----")
print("# accuracy :", accuracy)
print("# precision :", accuracy)
print("# recall :", recall)


'''
print('パターンマッチングスコアの精度')
print("# t1_p :", num_t1_p)
print("# f1_p :", num_f1_p)
print("# t0_p :", num_t0_p)
print("# f0_p :", num_f0_p)
print("# accuracy_p :", accuracy_p)
print("# F1_score_p :", F1_score_p)
'''

df = pd.read_csv("/Users/satogo/Library/CloudStorage/GoogleDrive-gogogou0110@gmail.com/マイドライブ/卒業研究　コード(AIX  バックアップ)/MHA-BiLSTM (12 16)/pattern_matching_score(bert_sentence)_last.csv", sep=',')
df["立場ラベル(予測)"] = labels_predicted.tolist()
df["立場ラベル(予測)"] = df["立場ラベル(予測)"].fillna(0).astype(np.int64)
df.to_csv('pattern_matching_score(bert_sentence).csv', index=None)

#print('格納しました')