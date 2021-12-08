import sys

sys.path.append('D:/Projects_LEO/Project_NLP_LEO/chatbot_LEO/')
from utils.Preprocess import Preprocess
from models.intent.IntentModel import IntentModel

p = Preprocess(word2index_dic='./chatbot_LEO/train_tools/dict/chatbot_dict.bin',
               userdic='./chatbot_LEO/utils/user_dic.tsv')

intent = IntentModel(model_name='./chatbot_LEO/models/intent/intent_model.h5', proprocess=p)
query = "오늘 탕수육 주문 가능한가요?"
predict = intent.predict_class(query)
predict_label = intent.labels[predict]

print(query)
print("의도 예측 클래스 : ", predict)
print("의도 예측 레이블 : ", predict_label)

