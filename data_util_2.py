import pandas as pd  # 데이터 읽기에 사용하는 라이브러리
from datasets import Dataset, DatasetDict  # 딥러닝에서 입력 데이터를 관리하는 라이브러리


# 감성 대화 말뭉치 학습 데이터 집합 읽어오기
train_df = pd.read_excel('ai_practice/dataset/sentiment_train.xlsx')

# 감성 대화 말뭉치 검증 데이터 집합 읽어오기
test_df = pd.read_excel('ai_practice/dataset/sentiment_val.xlsx')

# 감성 대화 말뭉치의 감성 분류 클래스
sentiments = ['분노', '기쁨', '불안', '당황', '슬픔', '상처']

# 감성 클래스를 one hot encoding의 labels로 변환
train_df['labels'] = train_df['감정_대분류'].map(lambda s: [1. if sentiments.index(s) == idx else 0. for idx in range(len(sentiments))])
test_df['labels'] = test_df['감정_대분류'].map(lambda s: [1. if sentiments.index(s) == idx else 0. for idx in range(len(sentiments))])

# 딥러닝 입력 데이터 집합으로 변환
train_ds = Dataset.from_pandas(train_df[['사람문장1', '감정_대분류', 'labels']], split='train')
test_ds = Dataset.from_pandas(test_df[['사람문장1', '감정_대분류', 'labels']], split='test')

dataset = DatasetDict()
dataset['train'] = train_ds
dataset['test'] = test_ds