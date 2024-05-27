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
import nltk
#nltk.download('punkt')
from nltk.tokenize import word_tokenize
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






dataset_path = rootDir="/Users/satogo/Downloads/研究/皮肉/Sarcasm-Detection-master/データセット/data2_preprocessed.csv"
sentiment_dict_path = '/Users/satogo/Downloads/研究/皮肉/Sarcasm-Detection-master/SVM/Bush/wordlist_japanese.csv'

tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking').tokenize

sentiment_words = []
sentiment_words_hiragana = []
sentiment_scores = []
df = pd.read_csv(sentiment_dict_path, sep=',')
for line in df.values:
   sentiment_words.append(line[0]) #辞書に含まれるスコアつきの単語(漢字)
   sentiment_words_hiragana.append(line[1]) #辞書に含まれるスコアつきの単語(ひらがなのみ)
   sentiment_scores.append(line[3]) #辞書に含まれたスコア
   

#print(sentiment_words[1])

review_texts = []
labels = []
ids = []
tokens_list = []

df = pd.read_csv(dataset_path, sep=',')
for line in df.values:
   token_list = []
   review_text = line[0]   # テキストは1列目
   review_texts.append(review_text)
   # テキストをトークン化し、トークンをリストに格納
   token_list = mecab.parse(review_text).split(" ")
   del token_list[-1] ##最後に'\n'が入っちゃってるからそれを削除
   label = line[1] #ラベルは2列目
   labels.append(label)
   id = line[2] #idは3列目
   ids.append(id)

   print(token_list)

   ##words = words.split('　')
   tokens_list.append(token_list) #トークン化した文章の各単語を配列に格納


sentiment_words_list_list = []  #トークン化されたテキスト中の各単語を配列に格納
sentiment_scores_list_list = [] #トークン化されたテキスト中の各単語の感情スコアを配列に格納

paradox_words = ["逆", "ぎゃく" "さすが", "流石", "すごい", "すごいな", "凄い", "凄いな", "すげえ","すげぇ", "すげー", "むしろ", "寧ろ", "方", "ほう","良く", "よく", "w", "笑", "?","😂"]
final_words = ["w", "笑", "?", "😂"]
paradox_scores_list_list = []


max_sentiment_words_number = 0
max_paradox_words_number = 0
paradox_words_position = 0
pattern_matching_list = [] #パターンマッチングしているかどうかを0or1で判断する配列
final_words_score_list = [] #文末単語が指定単語で終わっているかどうかを0or1で判断する配列

for tokens in tokens_list:
    sentiment_words_number = 0
    paradox_words_number = 0
    sentiment_words_list = []  #あるテキストに対しててトークン化されたテキスト中の各単語を配列に格納
    sentiment_scores_list = []  #あるテキストに対してトークン化されたテキスト中の各単語の感情スコアを配列に格納
    paradox_scores_list =[] #逆説になるような単語を含んでいるかどうかを判断する配列
    pattern_matching = 0
    paradox_words_position = -1 #oの位置に逆説単語がある場合を考慮して逆説単語がない間は値を-1に
    i = 0
    for token in tokens:
        sentiment_word = token
        result = classifier(token)[0]
        sentiment_score = result["score"]
        if result["label"] == "NEUTRAL":
            sentiment_score = 0
        elif result["label"] == "NEGATIVE":
            sentiment_score = -result["score"]

        if result["label"] == "POSITIVE" and i > paradox_words_position and paradox_words_position != -1:
            sentiment_word = token
            #pattern_matching = 1 
            pattern_matching = result["score"]

        sentiment_words_list.append(sentiment_word)
        if sentiment_score != 0: #感情スコアが0だったらリストには格納しない
            sentiment_scores_list.append(sentiment_score)
            sentiment_words_number += 1

        if token in paradox_words: #逆説の単語が含まれていたら
            paradox_score = 1
            paradox_words_position = i #逆説が含まれている単語の位置を特定
        else:
            paradox_score = 0
        paradox_scores_list.append(paradox_score)
        paradox_words_number += 1 
        i += 1

    if sentiment_words_number > max_sentiment_words_number: #感情有りの単語数の最大長を調べる
        max_sentiment_words_number = sentiment_words_number
    if paradox_words_number > max_paradox_words_number: #トークン化した単語数の最大長を調べる
        max_paradox_words_number = paradox_words_number
    sentiment_words_list_list.append(sentiment_words_list)
    sentiment_scores_list_list.append(sentiment_scores_list)
    paradox_scores_list_list.append(paradox_scores_list)
    pattern_matching_list.append(pattern_matching)
    if sentiment_words_list[-1] in final_words: #文末単語が指定単語で終わっていたら1を、それ以外は0を代入
        final_words_score_list.append(1)
    else:
        final_words_score_list.append(0)

print("\n")

for i in sentiment_scores_list_list:    
    print(i)
    print("\n")
    
# 出力結果をCSVに保存
#ファイル出力

shift_columns = 1
with open("文章中に含まれる各単語の感情スコア(bert).csv", "w", encoding="utf-8", newline="") as f: #csvファイルに書く
    writer = csv.writer(f)
    header = []
    for i in range(1 + max_sentiment_words_number + max_paradox_words_number + 3 + 2 + 2 + 4): #必要な数だけ列を確保
        header.append("")  # 空白の列を挿入
    header[0] = "コメント"
    header[1] = ""
    header[2] = "感情スコア"
    i = 3
    while(i < max_sentiment_words_number+3):
        header[i] = ""
        i += 1
    header[i] = "逆説スコア"
    i+=1
    while(i < max_sentiment_words_number+3+max_paradox_words_number):
        header[i] = ""
        i += 1
    i+=1
    header[i] = "パターンマッチング"
    header[i+1] = ""
    header[i+2] = "文末単語"
    header[i+3] = ""
    header[i+4] = "正解ラベル"
    header[i+5] = ""
    header[i+6] = "id"
    writer.writerow(header)

    # 各行のデータをずらして書き込む
    for c, scores_list, paradox_score_list, pattern_matching, final_words, label, id in zip(review_texts, sentiment_scores_list_list, paradox_scores_list_list, pattern_matching_list, final_words_score_list, labels, ids):
        shifted_row = [c] + [None] * shift_columns + scores_list + [None] *(max_sentiment_words_number - len(scores_list) + 1)  + paradox_score_list + [None] *(max_paradox_words_number - len(paradox_score_list) + 1) + [pattern_matching] + [None] * shift_columns + [final_words] + [None]+ [label] + [None] + [id]
        writer.writerow(shifted_row)
        #for paradox_score in paradox_score_list:
        #    if paradox_score == 1:

