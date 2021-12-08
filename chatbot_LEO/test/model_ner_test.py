import sys
sys.path.append('D:/Projects_LEO/Project_NLP_LEO/chatbot_LEO/')
from utils.Preprocess import Preprocess
from models.ner.NerModel import NerModel

p = Preprocess(word2index_dic='./chatbot_LEO/train_tools/dict/chatbot_dict.bin',
               userdic='./chatbot_LEO/utils/user_dic.tsv')


ner = NerModel(model_name='./chatbot_LEO/models/ner/ner_model.h5', proprocess=p)
query = '오늘 오전 13시 2분에 탕수육 주문 하고 싶어요'
predicts = ner.predict(query)
tags = ner.predict_tags(query)
print(predicts)
print(tags)

