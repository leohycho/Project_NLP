import sys, os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils.Preprocess import Preprocess

# from tensorflow.keras import preprocessing

sent = "내일 오전 10시에 짬뽕 주문하고 싶어ㅋㅋ"

# 전처리 객체 생성
p = Preprocess(userdic = './chatbot_LEO/utils/user_dic.tsv')

# p = Preprocess(word2index_dic='../train_tools/dict/chatbot_dict.bin',
#                userdic = '../utils/user_dic.tsv')

pos = p.pos(sent)

keywords = p.get_keywords(pos, without_tag=False)
print(keywords)

keywords = p.get_keywords(pos, without_tag=True)
print(keywords)

# w2i = p.get_wordidx_sequence(keywords)
# sequences = [w2i]
#
# MAX_SEQ_LEN = 15    # 임베딩 벡터 크기
# padded_seqs = preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post')
#
# print(keywords)
# print(sequences)
# print(padded_seqs)
