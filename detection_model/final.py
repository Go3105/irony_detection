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

'''

#use_cuda = torch.cuda.is_available() and True
#device = torch.device("cuda" if use_cuda else "cpu")
torch.set_default_tensor_type('torch.cuda.FloatTensor')
#torch.device("cuda")

# 日本語の事前学習モデル
#MODEL_NAME = '/home0/y2020/s2010320/Sarcasm_detection/MHA-BiLSTM/model_transformers_stance_manual/data/student/s2010320/Sarcasm_detection/MHA-BiLSTM/model/epoch=294-step=6490.ckpt'
MODEL_NAME = '/home0/y2020/s2010320/Sarcasm_detection/MHA-BiLSTM/model_transformers_stance_manual/data/student/s2010320/Sarcasm_detection/MHA-BiLSTM/model/0.7739130258560181' #aacuracy : 0.8547826086956521

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

df = pd.read_csv("/home0/y2020/s2010320/Sarcasm_detection/MHA-BiLSTM/pattern_matching_score(bert_sentence).csv", sep=',')
for line in df.values:
    review_text = line[0]
    label = line[13]-2
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

print("# scores:")
print(scores.size())
print("# predicted labels:")
print(labels_predicted)
print("# accuracy:")
print(accuracy)


print(labels_predicted.tolist())

df = pd.read_csv("/home0/y2020/s2010320/Sarcasm_detection/MHA-BiLSTM/pattern_matching_score(bert_sentence).csv", sep=',')
df["立場ラベル(予測)"] = labels_predicted.tolist()
df["立場ラベル(予測)"] = df["立場ラベル(予測)"].fillna(0).astype(np.int64)
df.to_csv('pattern_matching_score(bert_sentence).csv', index=None)



######################################################

'''

#use_cuda = torch.cuda.is_available() and True
#device = torch.device("cuda" if use_cuda else "cpu")
#torch.device("cuda")
#from livelossplot import PlotLosses

# 日本語の事前学習モデル
MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'

tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)
bert_sc = BertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
)
bert_sc = bert_sc.cuda()


# トークナイザのロード
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)

# 各データの形式を整える
max_length = 128
dataset_for_loader = []
pattern_matching_scores_list = []

df = pd.read_csv("/home0/y2020/s2010320/Sarcasm_detection/MHA-BiLSTM/pattern_matching_score(bert_sentence).csv", sep=',')
for line in df.values:
    review_text = line[0]
    #print(review_text) 表示してみる
    encoding = tokenizer(
        review_text,
        max_length=max_length,
        padding='max_length',
        truncation=True
    )
    encoding['labels'] = line[8] # ラベルを追加
    if line[14] == 0:
        encoding['stance'] = 0.85
    elif line[14] == 1:
        encoding['stance'] = 0.15
    encoding['pattern_matching_score'] = line[6]
    encoding = { k: torch.tensor(v) for k, v in encoding.items() }
    dataset_for_loader.append(encoding)
    pattern_matching_scores_list.append(line[6])
    #print(dataset_for_loader)
#print(pattern_matching_scores_list)

#二つのリストの統合
#p = list(zip(dataset_for_loader, pattern_matching_scores_list))

# データセットの分割 6 : 2 : 2
random.shuffle(dataset_for_loader) # ランダムにシャッフル
#dataset_for_loader, pattern_matching_scores_list = zip(*p)
n = len(dataset_for_loader)
n_train = int(0.6*n)
n_val = int(0.2*n)
dataset_train = dataset_for_loader[:n_train] # 学習データ
dataset_val = dataset_for_loader[n_train:n_train+n_val] # 検証データ
dataset_test = dataset_for_loader[n_train+n_val:] # テストデータ

# データセットからデータローダを作成
# 学習データはshuffle=Trueにする。
dataloader_train = DataLoader(
    dataset_train, batch_size=32, shuffle=True
)
dataloader_val = DataLoader(dataset_val, batch_size=64)   #256から32に変えた
dataloader_test = DataLoader(dataset_test, batch_size=64) #256から32に変えた

# 6-14
class BertForSequenceClassification_pl(pl.LightningModule):

    def __init__(self, model_name, num_labels, lr):
        # model_name: Transformersのモデルの名前
        # num_labels: ラベルの数
        # lr: 学習率

        super().__init__()

        # 引数のnum_labelsとlrを保存。
        # 例えば、self.hparams.lrでlrにアクセスできる。
        # チェックポイント作成時にも自動で保存される。
        self.save_hyperparameters()

        # BERTのロード
        self.bert_sc = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )

    # 学習データのミニバッチ(`batch`)が与えられた時に損失を出力する関数を書く。
    # batch_idxはミニバッチの番号であるが今回は使わない。
    def training_step(self, batch, batch_idx):
        pattern_matching_scores = batch.pop('pattern_matching_score')
        stance = batch.pop('stance')
        output = self.bert_sc(**batch)
        loss = output.loss

        #labels = batch.pop('labels') #精度のログをとる
        #labels_predicted = output.logits.argmax(-1)
        #num_correct = ( labels_predicted == labels ).sum().item()
        #num_predicted_1 = ( labels_predicted == 1 ).sum().item()
        #num_tp = ((labels_predicted == 1) & (labels_predicted == labels) ).sum().item()
        #accuracy = num_correct/labels.size(0) #精度
        #if num_predicted_1 > 0:
        #   precision = num_tp/ num_predicted_1
        #else:
        #   precision = 0
        self.log('train_loss', loss) # 損失を'train_loss'の名前でログをとる。
        #self.log('train_accuracy', float(accuracy))
        #self.log('train_precision', float(precision))
        return loss

    # 検証データのミニバッチが与えられた時に、
    # 検証データを評価する指標を計算する関数を書く。
    def validation_step(self, batch, batch_idx):
        pattern_matching_scores = batch.pop('pattern_matching_score')
        stance = batch.pop('stance')
        output = self.bert_sc(**batch)
        val_loss = output.loss

        labels = batch.pop('labels') #精度のログをとる
        #pattern_matching_scores = batch.pop('pattern_matching_score')
        labels_predicted = output.logits.argmax(-1)
        num_correct = ( labels_predicted == labels ).sum().item()
        num_predicted_1 = ( labels_predicted == 1 ).sum().item()
        num_tp = ((labels_predicted == 1) & (labels_predicted == labels) ).sum().item()
        accuracy = num_correct/labels.size(0) #精度
        if num_predicted_1 > 0:
           precision = num_tp/ num_predicted_1
        else:
           precision = 0
        self.log('val_accuracy', float(accuracy))
        self.log('val_precision', float(precision))
        self.log('val_loss', val_loss)# 損失を'val_loss'の名前でログをとる。

    # テストデータのミニバッチが与えられた時に、
    # テストデータを評価する指標を計算する関数を書く。
    def test_step(self, batch, batch_idx):
        labels = batch.pop('labels') # バッチからラベルを取得
        pattern_matching_scores = batch.pop('pattern_matching_score')
        stance = batch.pop('stance')
        output = self.bert_sc(**batch)
        labels_predicted = output.logits.argmax(-1)
        

        #パターンマッチングなし
        num_correct_no = ( labels_predicted == labels ).sum().item()
        num_predicted_1_no = ( labels_predicted == 1 ).sum().item() #1と予測した数
        num_tp_no = ((labels_predicted == 1) & (labels_predicted == labels) ).sum().item() #1と予測して正解だった数
        accuracy_no = num_correct_no/labels.size(0) #正解率
        if num_predicted_1_no > 0: #適合率
            precision_no = num_tp_no/ num_predicted_1_no
        else:
            precision_no = 0

        num_t1_no = (labels == 1).sum().item() #正解ラベルが1である数の合計
        if num_t1_no > 0:#再現率
            recall_no = num_tp_no / num_t1_no
        else:
            recall_no = 0
        
        global result
        if batch_idx == 0:
            result = 0
        if precision_no == 0 and recall_no == 0: #F1スコア
            F1_score_no = 0
        else:
            F1_score_no = (2*precision_no*recall_no)/(precision_no + recall_no) 
        result += F1_score_no
        self.log('test_accuracy_no', accuracy_no) # 精度を'accuracy'の名前でログをとる。
        self.log('test_precision_no', float(precision_no))
        self.log('test_recall_no', float(recall_no))
        self.log('test_F1score_no', float(F1_score_no))
        self.log('test_正解数_no',float( num_correct_no)) # 精度を'accuracy'の名前でログをとる。
        self.log('test_データ数_no', float(labels.size(0)))
        self.log('test_ラベル1と予測された数_no', float(num_predicted_1_no))
        self.log('test_正解ラベルが1である数_no', float(num_t1_no))
        if batch_idx == 3:
            result = result/4
            print('パターンマッチングなし')
            print('test_accuracy', accuracy_no) # 精度を'accuracy'の名前でログをとる。
            print('test_precision', float(precision_no))
            print('test_recall', float(recall_no))
            print('test_F1score', float(F1_score_no))
            print('test_正解数 ',float( num_correct_no)) # 精度を'accuracy'の名前でログをとる。
            print('test_データ数', float(labels.size(0)))
            print('test_ラベル1と予測された数', float(num_predicted_1_no))
            print('test_正解ラベルが1である数', float(num_t1_no))
            print('')


        #パターンマッチングあり
        #if pattern_matching_scores > 0: #パターンマッチングと組み合わせる
        #    labels_predicted = 1
        #print(labels_predicted)
        num_correct = (((labels_predicted == labels) & (labels_predicted + pattern_matching_scores == 0)) | ((pattern_matching_scores > 0) & (pattern_matching_scores == labels)) | ((labels_predicted == 1) & (labels_predicted == labels))).sum().item()
        num_predicted_1 = (( labels_predicted == 1 ) | (pattern_matching_scores > 0)).sum().item() #1と予測した数
        num_tp = (((labels_predicted == 1) & (labels_predicted == labels)) | ((pattern_matching_scores > 0) & (pattern_matching_scores == labels) )).sum().item() #1と予測して正解だった数
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
        if batch_idx == 3:
            print('パターンマッチングあり')
            print('test_accuracy', accuracy) # 精度を'accuracy'の名前でログをとる。
            print('test_precision', float(precision))
            print('test_recall', float(recall))
            print('test_F1score', float(F1_score))
            print('test_正解数 ',float( num_correct)) # 精度を'accuracy'の名前でログをとる。
            print('test_データ数', float(labels.size(0)))
            print('test_ラベル1と予測された数', float(num_predicted_1))
            print('test_正解ラベルが1である数', float(num_t1))
        self.log('test_accuracy', accuracy) # 精度を'accuracy'の名前でログをとる。
        self.log('test_precision', float(precision))
        self.log('test_recall', float(recall))
        self.log('test_F1score', float(F1_score))
        self.log('test_正解数 ',float( num_correct)) # 精度を'accuracy'の名前でログをとる。
        self.log('test_データ数', float(labels.size(0)))
        self.log('test_ラベル1と予測された数', float(num_predicted_1))
        self.log('test_正解ラベルが1である数', float(num_t1))

    # 学習に用いるオプティマイザを返す関数を書く。
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    


# 6-15
# 学習時にモデルの重みを保存する条件を指定
checkpoint = pl.callbacks.ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_top_k=1,
    save_weights_only=True,
    dirpath='model/',
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience = 25,
)

# 学習の方法を指定
trainer = pl.Trainer(
    gpus=1,
    max_epochs=10000,  
    callbacks = [checkpoint, early_stopping]
)


# 6-16
# PyTorch Lightningモデルのロード
model = BertForSequenceClassification_pl(
    MODEL_NAME, num_labels=2, lr=1e-6
)


# ファインチューニングを行う。
trainer.fit(model, dataloader_train, dataloader_val)
#tf.summary.scalar('loss', loss)

# 6-17
best_model_path = checkpoint.best_model_path # ベストモデルのファイル
print('ベストモデルのファイル: ', checkpoint.best_model_path)
print('ベストモデルの検証データに対する損失: ', checkpoint.best_model_score)

# 6-19
test = trainer.test(dataloaders=dataloader_test)
#print(f'Accuracy: {test[0]["accuracy"]:.2f}')

model = BertForSequenceClassification_pl.load_from_checkpoint(
    best_model_path
) 

# Transformers対応のモデルを./model_transformesに保存
if result > 0.6:
    model.bert_sc.save_pretrained('./model_final_F1/' +  str(result))
    print('-----------'+str(result)+'に保存しました'+'-----------')