# tensorboard --logdir=lightning_logs
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
import MeCab
import torch.nn.functional as F



#use_cuda = torch.cuda.is_available() and True
#device = torch.device("cuda" if use_cuda else "cpu")
torch.set_default_tensor_type('torch.cuda.FloatTensor')
#torch.device("cuda")

# 日本語の事前学習モデル
MODEL_NAME = '/home0/y2020/s2010320/Sarcasm_detection/MHA-BiLSTM/model_final_F1/0.6666666666666666' #final.py best
#MODEL_NAME = '/home0/y2020/s2010320/Sarcasm_detection/MHA-BiLSTM/model_final_F1/0.8142857142857143' #final.py try
#MODEL_NAME = '/home0/y2020/s2010320/Sarcasm_detection/MHA-BiLSTM/model_final_F1/0.742063492063492' #final.py(助詞のみ除去)

#MODEL_NAME = '/home0/y2020/s2010320/Sarcasm_detection/MHA-BiLSTM/model_chap6/0.753030303030303'  #Chap6.py best
#MODEL_NAME = '/home0/y2020/s2010320/Sarcasm_detection/MHA-BiLSTM/model_chap6/0.7678571428571429'  #Chap6.py try

tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
bert_sc = BertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
)
bert_sc = bert_sc.cuda()

# 各データの形式を整える
max_length = 128
dataset_for_loader = []

text_list =[]
label_list = []



#品詞による前処理
def get_removed_text(text):
    mecab = MeCab.Tagger("-Owakati")  # 分かち書き
    parsed_text = mecab.parse(text)

    mecab = MeCab.Tagger()
    node = mecab.parseToNode(parsed_text)

    words = []
    while node:
        if node.feature.split(',')[0] != 'BOS/EOS':
            word = node.surface
            pos = node.feature.split(',')[0]
            pos_kind = node.feature.split(',')[1]
            #助詞は除去
            #if (pos == '助詞' or pos == '助動詞'):
            if (pos == '助詞'):
                pass
            #名詞かつ「固有名詞または代名詞」は除去 pos_kind == '固有名詞' or 
            #elif (pos == '名詞' and (pos_kind == '代名詞')):
            #    #print(word)
            #    pass
            #それ以外の品詞だった場合文に繋げる
            else:
                #mecab = MeCab.Tagger("-Ochasen") 
                #lemmatized_word_node = mecab.parseToNode(word)[1]
                #print(lemmatized_word_node)
                words.append(word)
        node = node.next
    removed_text = ''.join(words)
    return removed_text





df = pd.read_csv("/home0/y2020/s2010320/Sarcasm_detection/MHA-BiLSTM/pattern_matching_score(bert_sentence).csv", sep=',')
for line in df.values:
    review_text = line[0]
    label = line[8]
    text_list.append(review_text)
    label_list.append(label)

encoding = tokenizer(
    text_list,
    max_length=max_length,
    padding='max_length',
    truncation=True
)
encoding = { k: torch.tensor(v) for k, v in encoding.items() }
labels = torch.tensor(label_list).cuda()

# 推論
with torch.no_grad():
    output = bert_sc.forward(**encoding)
scores = output.logits # 分類スコア
labels_predicted = scores.argmax(-1) # スコアが最も高いラベル
num_correct = (labels_predicted==labels).sum().item() # 正解数
accuracy = num_correct/labels.size(0) # 精度
num_predicted_1 = ( labels_predicted == 1 ).sum().item()
num_tp = ((labels_predicted == 1) & (labels_predicted == labels) ).sum().item()
num_predicted_1 = ( labels_predicted == 1 ).sum().item() #1と予測した数
num_tp = ((labels_predicted == 1) & (labels_predicted == labels) ).sum().item() #1と予測して正解だった数
accuracy = num_correct/labels.size(0) #正解率
if num_predicted_1 > 0: #適合率
    precision = num_tp/ num_predicted_1
else:
    precision = 0

num_t1 = (labels == 1).sum().item() #正解ラベルが1である数の合計
if num_t1 > 0:#再現率
    recall = num_tp / num_t1
else:
    recall = 0

if precision == 0 and recall == 0: #F1スコア
    F1_score = 0
else:
    F1_score = (2*precision*recall)/(precision + recall) 

#print("# scores :")
#print(scores.size())
#print("# predicted labels :")
#print(labels_predicted)
print("既存BERTのみのスコア")
print(" accuracy : ", accuracy)
print(" precision : ", precision)
print(" recall : ", recall)
print(" F1_score : ", F1_score)
#print(F1_score)

#df = pd.read_csv("/home0/y2020/s2010320/Sarcasm_detection/MHA-BiLSTM/pattern_matching_score(bert_sentence).csv", sep=',')
#df["最終予測ラベル"] = labels_predicted.tolist()
#df["最終予測ラベル"] = df["最終予測ラベル"].fillna(0).astype(np.int64)
#df.to_csv('pattern_matching_score(bert_sentence).csv', index=None)

#print('格納しました')





i = 0
a = 0
b = 0
k = 0
sum_x0 = 0
sum_x1 = 0

scores.tolist()
final_predict_labels = labels_predicted.tolist()

df = pd.read_csv("/home0/y2020/s2010320/Sarcasm_detection/MHA-BiLSTM/pattern_matching_score(bert_sentence).csv", sep=',')
for line in df.values:
    #if labels_predicted.tolist()[i] == 0 and line[8] == 1:
    #    print('a : ',(line[12], line[14]))
    #elif labels_predicted.tolist()[i] == 1 and line[8] == 0:
    #    print('b :', (line[12], line[14]))
    #if line[12] == 2 and line[14] == 1 and labels_predicted.tolist()[i] == 1:
    #    print('c :', line[8])
    #if line[12] == 3 and line[14] == 1:
    #    print('d :', (labels_predicted.tolist()[i], line[8]))
    x = scores[i]
    if line[12] == 1 and line[14] == 1 : #bertではポジティブ、2値分類ではネガティブと分類された時
        #if (labels_predicted.tolist()[i] == 0) and line[16] > 0.9: #予測ラベルが0かつ既存bertによる感情スコアが0.9以上の時

        #下3行はどれが良いか悩み中
        #if (labels_predicted.tolist()[i] == 0) and line[16] > 0.9 and float(x[0]-x[1]) < 2: ########################### float(x[0]-x[1]) < 2　とりあえず2にした
        #if (labels_predicted.tolist()[i] == 0) and line[16] > 0.9 and float(x[0]+x[1]) < -0.6: ########################### float(x[0]-x[1]) < 2
        if (labels_predicted.tolist()[i] == 0) and line[16] > 0.9 and (float(x[0]+x[1]) < -0.6 or float(x[0]-x[1]) < 2) : 
            #print(line[0])
            #print('{:.3f}'.format(float(x[0]+x[1])), line[16], final_predict_labels[i] ,line[8])
            #print(float('{:.3f}'.format(float(x[0]+x[1]))), float('{:.3f}'.format(float(x[0]-x[1]))), line[16], final_predict_labels[i] ,line[8])
            #print(x, line[16], final_predict_labels[i] ,line[8])
            final_predict_labels[i] = 1
        
    ##パターンマッチング
    if line[6] > 0 and line[14] == 1: #パターんマッチングスコア　> 0 かつ 感情2値分類の結果がネガティブなら
        final_predict_labels[i] = 1 #予測ラベルを1に
    
    i += 1


##最終精度検証
i = 0
tp = 0
fp = 0
tn = 0
fn = 0
df = pd.read_csv("/home0/y2020/s2010320/Sarcasm_detection/MHA-BiLSTM/pattern_matching_score(bert_sentence).csv", sep=',')
for line in df.values:
    if line[8] == 1 and final_predict_labels[i] == 1:
        tp += 1
    elif line[8] == 0 and final_predict_labels[i] == 1:
        fp += 1
    elif line[8] == 0 and final_predict_labels[i] == 0:
        tn += 1
    elif line[8] == 1 and final_predict_labels[i] == 0:
        fn += 1
    i+=1

final_accuracy = (tp + tn)/(tp + fp + tn + fn)
final_precision = tp/(tp + fp)
final_recall = tp/(tp + fn)
final_F1 = (2*final_precision*final_recall)/(final_precision + final_recall)

print()
print('=====最終スコア(既存BERT + 提案手法)=====')
print(" final_accuracy : ", final_accuracy)
print(" final_precision : ", final_precision)
print(" final_recall : ", final_recall)
print(" final_F1_score : ", final_F1)