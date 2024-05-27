from transformers import pipeline, AutoModelForSequenceClassification, BertJapaneseTokenizer,BertTokenizer, BertForSequenceClassification

# パイプラインの準備
model = AutoModelForSequenceClassification.from_pretrained("koheiduck/bert-japanese-finetuned-sentiment")
#model = AutoModelForSequenceClassification.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")  
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
classifier = pipeline("sentiment-analysis",model=model,tokenizer=tokenizer)


result = classifier("黙ってて")[0]
print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

result = classifier("低評価0万突破おめでたい")[0]
print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
