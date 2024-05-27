from transformers import pipeline, AutoModelForSequenceClassification, BertJapaneseTokenizer,BertTokenizer, BertForSequenceClassification

import csv
import pandas as pd
import numpy as np
import MeCab
# パイプラインの準備
model = AutoModelForSequenceClassification.from_pretrained("koheiduck/bert-japanese-finetuned-sentiment")
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
classifier = pipeline("sentiment-analysis",model=model,tokenizer=tokenizer)

sentiment_scores_list = []
sentiment_scores_list2 = []
sentiment_scores_list3 = [] #3値を2値に(ラベル1→3)
sentiment_values_list = [] 
b_true_1 = 0
b_false_2 = 0
b_false_3 = 0
m_true_1 = 0
m_false_2 = 0
m_false_3 = 0

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
            else:
                words.append(word)
        node = node.next
    removed_text = ''.join(words)
    return removed_text


df = pd.read_csv("/home0/y2020/s2010320/Sarcasm_detection/MHA-BiLSTM/pattern_matching_score(bert_sentence).csv", sep=',')
for line in df.values:
    review_text = line[0]
    actually_label = line[8]
    stance_bert_label = line[12]
    if (actually_label == 1) & (stance_bert_label == 1):
        b_true_1 += 1
    elif (actually_label == 1) & (stance_bert_label == 2):
        b_false_2 += 1
    elif (actually_label == 1) & (stance_bert_label == 3):
        b_false_3 += 1

    stance_manual_label = line[11]
    if (actually_label == 1) & (stance_manual_label == 1):
        m_true_1 += 1
    elif (actually_label == 1) & (stance_manual_label == 2):
        m_false_2 += 1
    elif (actually_label == 1) & (stance_manual_label == 3):
        m_false_3 += 1
    sentiment_scores_list.append(stance_bert_label)
    sentiment_scores_list2.append(stance_manual_label)

    if stance_manual_label == 1:
        a = 3
    else:
        a = stance_manual_label
    sentiment_scores_list3.append(a)

print("BERT: b_true_1 :", b_true_1)
print("BERT: b_false_2 : ", b_false_2)
print("BERT: b_false_3 : ", b_false_3)

print()

print("BERT: m_true_1 :", m_true_1)
print("BERT: m_false_2 : ", m_false_2)
print("BERT: m_false_3 : ", m_false_3)


#df = pd.read_csv("/home0/y2020/s2010320/Sarcasm_detection/MHA-BiLSTM/pattern_matching_score(bert_sentence).csv", sep=',')
#df["立場ラベル(手動)"] = sentiment_scores_list2
#df["立場ラベル(手動)"] = df["立場ラベル(手動)"].fillna(0).astype(np.int64)
#df["立場ラベル(bert)"] = sentiment_scores_list
#df["立場ラベル(bert)"] = df["立場ラベル(bert)"].fillna(0).astype(np.int64)
#df["立場ラベル(手動(2値))"] = sentiment_scores_list3
#df["立場ラベル(手動(2値))"] = df["立場ラベル(手動(2値))"].fillna(0).astype(np.int64)
#df["感情スコア(bert)"] = sentiment_values_list
#df["感情スコア(bert)"] = df["感情スコア(bert)"].fillna(0).astype(np.int64)
#df.to_csv('pattern_matching_score(bert_sentence).csv', index=None)

