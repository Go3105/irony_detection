from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertForSequenceClassification

import tensorflow as tf
import pytorch_lightning as pl

import csv
import pandas as pd
import numpy as np
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from transformers import pipeline, AutoModelForSequenceClassification, BertJapaneseTokenizer,BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F


from preprocessing_japanese import preprocess

####################################### コメント入力、前処理 #######################################

text = preprocess().text_cleaning(input('コメントを入力して下さい > ')) 




####################################### 既存のBERTモデルで感情分析 #######################################

model = AutoModelForSequenceClassification.from_pretrained("koheiduck/bert-japanese-finetuned-sentiment")
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
classifier = pipeline("sentiment-analysis",model=model,tokenizer=tokenizer)
result = classifier(text)[0]  #コメントを入力
bert_sentiment_score  = round(result['score'], 4) #既存のBERTモデルにおける感情スコア
bert_sentiment_label = result['label'] #既存のBERTモデルにおける感情ラベル
if bert_sentiment_label == "POSITIVE":
    bert_sentiment_label = 1
elif bert_sentiment_label == "NEGATIVE":
    bert_sentiment_label = 2
else :
    bert_sentiment_label = 3




######################################## 感情スコアの2値分類 #######################################

MODEL_NAME = '/home0/y2020/s2010320/Sarcasm_detection/MHA-BiLSTM/model_transformers_stance_manual/0.8294481847503103' #感情スコアの2値分類モデル
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
bert_sc = BertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
)

# 各データの形式を整える
max_length = 128
dataset_for_loader = []

text_list =[]
label_list = []
text_list = [text]


encoding = tokenizer(
    text_list,
    max_length=max_length,
    padding='max_length',
    truncation=True
)
encoding = { k: torch.tensor(v) for k, v in encoding.items() }

# 推論
with torch.no_grad():
    output = bert_sc.forward(**encoding)
scores = output.logits # 分類スコア
labels_predicted_sentiment = scores.argmax(-1) # スコアが最も高いラベル






######################################## 皮肉の2値分類(既存手法のみ) #######################################

MODEL_NAME = '/home0/y2020/s2010320/Sarcasm_detection/MHA-BiLSTM/model_final_F1/0.6666666666666666' #final.py best

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
text_list = [text]

encoding = tokenizer(
    text_list,
    max_length=max_length,
    padding='max_length',
    truncation=True
)
encoding = { k: torch.tensor(v) for k, v in encoding.items() }
#label_list = torch.tensor(label_list).cuda()

# 推論
with torch.no_grad():
    output = bert_sc.forward(**encoding)
scores = output.logits # 分類スコア
labels_predicted = scores.argmax(-1)


######################################## 提案手法(感情、ラベル予測確率) #######################################

# ソフトマックス関数の適用
scores = F.softmax(scores, dim=1)
scores.tolist()
x = scores[0]
if bert_sentiment_label == 1 and labels_predicted_sentiment == 1 : #BERTによる感情分析の結果がポジティブかつファインチューニングした感情分析(2値分類)の結果がネガティブ
        if labels_predicted == 0 and bert_sentiment_score > 0.9 and float(x[0]) < 0.9: #ソフトマックスを使った時
            labels_predicted = 1




########################################  提案手法(パターンマッチングスコア) #######################################

#各単語の感情スコア算出にbertモデルを使用
from transformers import pipeline, AutoModelForSequenceClassification, BertJapaneseTokenizer,BertTokenizer, BertForSequenceClassification
from transformers import AutoTokenizer
# パイプラインの準備
model = AutoModelForSequenceClassification.from_pretrained("koheiduck/bert-japanese-finetuned-sentiment")
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
classifier = pipeline("sentiment-analysis",model=model,tokenizer=tokenizer)

tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking').tokenize

paradox_words = ["逆に", "ぎゃくに" "さすが", "流石", "むしろ", "寧ろ", "方が", "ほうが","良く", "よく", "低評価", "👎", "バッド", "Bad", "bad", "暴力"]
final_words = ["w", "笑", "?", "^ ^", "^ ^", "わろた", "ワロタ", "🤪", "😊", "😀", "😃", "😄", "😆", "😁", "😂", "🤣"]

sentence_score = 0 #とりあえずパターンマッチングスコアを０にしとく
final_words_score = 0 #とりあえず文末単語スコアを０にしとく
for keyword in paradox_words:
    if keyword in text: #逆説単語のうちいずれかが文に含まれていたら
        part_of_text = text.split(keyword, 1) #逆説単語の後を切り抜く
        #print(part_of_text[1])
        result = classifier(part_of_text[1])[0] #切り抜いた部分を感情分析
        if result["label"] == "POSITIVE": #切り抜いた部分がポジティブだったらパターンマッチングスコアを更新
            sentence_score = round(result["score"], 5) #round()関数を使って小数点以下の桁数を5桁に指定
for final_keyword in final_words: #文末単語のうちいずれかが与えられた文の文末単語と一致したら
    if final_keyword == text[-1]:
        final_words_score = 1
pattern_matching_score = sentence_score*final_words_score
if pattern_matching_score > 0 and labels_predicted_sentiment == 1: #パターンマッチングスコアが0以上かつ感情スコアが1(ネガティブ)だったら予測ラベルを1にする
    labels_predicted = 1

print()

if labels_predicted == 1:
    print('皮肉です !')
elif labels_predicted == 0:
    print('皮肉ではありません !')

print()