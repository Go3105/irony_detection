import re
import requests
import unicodedata

from bs4 import BeautifulSoup
import os
import urllib.request
import csv
import pandas as pd
from collections import Counter

from transformers import AutoTokenizer
import MeCab

#tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking').tokenize

class preprocess:
    def __init__(self):
        pass

    def clean_text(self, text):
        replaced_text = text.lower()
        replaced_text = re.sub(r'[【】]', '', replaced_text)       # 【】の除去
        replaced_text = re.sub(r'[（）()]', '', replaced_text)     # （）の除去
        replaced_text = re.sub(r'[［］\[\]]', '', replaced_text)   # ［］の除去
        replaced_text = re.sub(r'[@＠]\w+', '', replaced_text)  # メンションの除去
        replaced_text = re.sub(r'http?://[\w/:%#\$&\?\(\)~\.=\+\-]+', '', replaced_text) # URLの除去
        replaced_text = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-]+', '', replaced_text) # URLの除去
        replaced_text = re.sub(r'　', '', replaced_text)  # 全角空白の除去
        #replaced_text = lambda replaced_text: re.search(r'[ぁ-ん]+|[ァ-ヴー]+|[一-龠]+', replaced_text) # 日本語に限定
        replaced_text = re.sub(r'\n', '', replaced_text)
        replaced_text = re.sub(r'\r', '', replaced_text)
        replaced_text = re.sub(r'[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。、?！｀＋￥％]', '', replaced_text) #半角記号削除
        return replaced_text

    def clean_html_tags(self, html_text):
        soup = BeautifulSoup(html_text, 'html.parser')
        cleaned_text = soup.get_text()
        cleaned_text = ''.join(cleaned_text.splitlines())
        return cleaned_text

    def clean_html_and_js_tags(self, html_text):
        soup = BeautifulSoup(html_text, 'html.parser')
        [x.extract() for x in soup.findAll(['script', 'style'])]
        cleaned_text = soup.get_text()
        cleaned_text = ''.join(cleaned_text.splitlines())
        return cleaned_text

    def normalize_unicode(self, text, form='NFKC'):
        normalized_text = unicodedata.normalize(form, text)
        return normalized_text

    def normalize_number(self, text):
        replaced_text = re.sub(r'\d+', '0', text)
        return replaced_text

    def lower_text(self, text):
        return text.lower()

    def normalize(self, text):
        normalized_text = self.normalize_unicode(text)
        normalized_text = self.normalize_number(normalized_text)
        normalized_text = self.lower_text(normalized_text)
        return normalized_text

    def text_cleaning(self, text):
        text = self.clean_text(text)
        text = self.clean_html_tags(text)
        text = self.clean_html_and_js_tags(text)
        text = self.normalize(text)
        return text



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
            elif (pos == '名詞' and (pos_kind == '代名詞')):
                #print(word)
                pass
            #それ以外の品詞だった場合文に繋げる
            else:
                #mecab = MeCab.Tagger("-Ochasen") 
                #lemmatized_word_node = mecab.parseToNode(word)[1]
                #print(lemmatized_word_node)
                words.append(word)
        node = node.next
    removed_text = ''.join(words)
    return removed_text


# データの前処理・前処理したコメントの格納
cleaned_texts = []
final_cleaned_texts = []
labels = []
ids = []
id = 0

df = pd.read_csv("/home0/y2020/s2010320/Sarcasm_detection/MHA-BiLSTM/pattern_matching_score(bert_sentence).csv", sep=',')
for line in df.values:
   text = line[0] #コメント
   cleaned_text = preprocess().text_cleaning(text) #半角記号などの除去
   cleaned_texts.append(cleaned_text)



def get_pos_info(text):
    mecab = MeCab.Tagger("-Owakati")  # 分かち書きモード
    parsed_text = mecab.parse(text)

    mecab = MeCab.Tagger()
    node = mecab.parseToNode(parsed_text)

    pos_info = []
    while node:
        if node.feature.split(',')[0] != 'BOS/EOS':
            word = node.surface
            pos = node.feature.split(',')[0]
            pos_kind = node.feature.split(',')[1]
            #名詞以外、もしくは「名詞かつ「固有名詞または代名詞」」
            if (pos == '助詞' or pos == '助動詞') :
                pos_info.append((word, pos, ''))
            elif (pos == '名詞' and (pos_kind == '固有名詞' or pos_kind == '代名詞')):
                pos_info.append((word, pos, pos_kind))
        node = node.next
    return pos_info
