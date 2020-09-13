
from gensim.models.word2vec import Word2Vec
 
# 读取数据，用gensim中的word2vec训练词向量
file = open('sentence.txt')
sss=[]
while True:
    ss=file.readline().replace('\n','').rstrip()
    if ss=='':
        break
    s1=ss.split(" ")
    sss.append(s1)
file.close()
model = Word2Vec(size=200, workers=5,sg=1)  # 生成词向量为200维，考虑上下5个单词共10个单词，采用sg=1的方法也就是skip-gram
model.build_vocab(sss)
model.train(sss,total_examples = model.corpus_count,epochs = model.iter)
model.save('./data/gensim_w2v_sg0_model')            # 保存模型 
new_model = gensim.models.Word2Vec.load('w2v_model') # 调用模型
sim_words = new_model.most_similar(positive=['女人'])
for word,similarity in sim_words:
    print(word,similarity)                           # 输出’女人‘相近的词语和概率
