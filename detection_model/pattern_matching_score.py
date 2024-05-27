#逆説単語を見つけたらそれ以降の文を切り抜いてbertで感情分析を行い、
#パターンマッチングしているかを確かめる方法

import decimal
import numpy as np
import torch
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import re
import os
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
from torch.utils.data import DataLoader, Dataset, TensorDataset
#nltk.download('punkt')
import MeCab
from torch import tensor
from torch import int32, float32
from pprint import pprint
import time
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import dill
import pickle
from transformers import AutoTokenizer
import MeCab
mecab = MeCab.Tagger("-Owakati")


#各単語の感情スコア算出にbertモデルを使用
from transformers import pipeline, AutoModelForSequenceClassification, BertJapaneseTokenizer,BertTokenizer, BertForSequenceClassification

# パイプラインの準備
model = AutoModelForSequenceClassification.from_pretrained("koheiduck/bert-japanese-finetuned-sentiment")
#model = AutoModelForSequenceClassification.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")  
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
classifier = pipeline("sentiment-analysis",model=model,tokenizer=tokenizer)

#sentiment_dict_path = '/Users/satogo/Downloads/研究/皮肉/Sarcasm-Detection-master/SVM/Bush/wordlist_japanese.csv'

tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking').tokenize

review_texts = []
labels = []
ids = []

df = pd.read_csv("/home0/y2020/s2010320/Sarcasm_detection/MHA-BiLSTM/pattern_matching_score(bert_sentence).csv", sep=',')
for line in df.values:
   review_text = line[0]   # テキストは1列目
   review_texts.append(review_text)
   #label = line[1] #ラベルは2列目
   #labels.append(label)
   #id = line[2] #idは3列目
   #ids.append(id)

paradox_words = ["逆に", "ぎゃくに" "さすが", "流石", "むしろ", "寧ろ", "方が", "ほうが","良く", "よく", "低評価", "👎", "バッド", "Bad", "bad", "暴力"]
final_words = ["w", "笑", "?", "？", "^ ^", "^ ^", "わろた", "ワロタ", "🤪", "😊", "😀", "😃", "😄", "😆", "😁", "😂", "🤣"]
sentence_score_list = []
pattern_matching_score_list = []
final_words_score_list = []


for text in review_texts:
    sentence_score = 0 #とりあえずパターンマッチングスコアを０にしとく
    final_words_score = 0 #とりあえず文末単語スコアを０にしとく
    for keyword in paradox_words:
        if keyword in text: #逆説単語のうちいずれかが文に含まれていたら
            part_of_text = text.split(keyword, 1) #逆説単語の後を切り抜く
            #print(part_of_text[1])
            result = classifier(part_of_text[1])[0] #切り抜いた部分を感情分析
            if result["label"] == "POSITIVE": #切り抜いた部分がポジティブだったらパターンマッチングスコアを更新
                sentence_score = round(result["score"], 5) #round()関数を使って小数点以下の桁数を5桁に指定
    sentence_score_list.append(sentence_score)
    for final_keyword in final_words: #文末単語のうちいずれかが与えられた文の文末単語と一致したら
        if final_keyword == text[-1]:
            final_words_score = 1
    final_words_score_list.append(final_words_score)
    pattern_matching_score = sentence_score*final_words_score
    pattern_matching_score_list.append(pattern_matching_score)
        
df = pd.read_csv("/home0/y2020/s2010320/Sarcasm_detection/MHA-BiLSTM/pattern_matching_score(bert_sentence).csv", sep=',')
df["パターンマッチング(合計)スコア"] = pattern_matching_score_list
#df["感情スコア(bert)"] = df["感情スコア(bert)"].fillna(0).astype(np.int64)
df.to_csv('pattern_matching_score(bert_sentence).csv', index=None)