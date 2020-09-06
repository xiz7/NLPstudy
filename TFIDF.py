# -*- coding: utf-8 -*-

import numpy as np
 
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.feature_extraction.text import TfidfTransformer
 
def sklearn_tfidf():
    tag_list = ['iphone guuci huawei watch huawei',
                'huawei watch iphone watch iphone guuci',
                'skirt skirt skirt flower',
                'watch watch huawei']
    
    vectorizer = CountVectorizer() #将文本中的词语转换为词频矩阵  
    X = vectorizer.fit_transform(tag_list) #计算个词语出现的次数
       
    transformer = TfidfTransformer()  
    tfidf = transformer.fit_transform(X)  #将词频矩阵X统计成TF-IDF值  
    print(tfidf.toarray())
    
def tfidf_alg():
    docs = np.array(['iphone guuci huawei watch huawei',
                     'huawei watch iphone watch iphone guuci',
                     'skirt skirt skirt flower',
                     'watch watch huawei'])
    
    words = np.array(['iphone', 'guuci', 'huawei', 'watch', 'skirt', 'flower'])
    #calc cf way1, 词在文档中出现的个数
    cfs = []
    for e in docs:
       cf = [e.count(word) for word in words]
       cfs.append(cf)
    print('cfs way1:\n', np.array(cfs))
    
    #calc cf way2, 词在文档中出现的个数
    cfs = []
    cfs.extend([e.count(word) for word in words] for e in docs)
    cfs = np.array(cfs)
    print('cfs way2:\n', cfs)
    
    #calc tf way1, 词在文档中的频率
    tfs = []
    for e in cfs:
        tf = e/(np.sum(e))
        tfs.append(tf)
    print('tfs way1:\n', np.array(tfs))
 
    #calc tf way2, 词在文档中的频率
    tfs = []
    tfs.extend(e/(np.sum(e)) for e in cfs)#不能使用append()
    print('tfs:\n',np.array(tfs))
    
    #calc df way1, 包含词的文档个数
    dfs = list(np.zeros(words.size, dtype=int))
    for i in range(words.size):
        for doc in docs:
            if doc.find(words[i]) != -1:
                dfs[i] += 1
    print('calc df way1:', dfs)
    
    #calc df way2, 包含词的文档个数
    dfs = []
    for i in range(words.size):
        oneHot = [(doc.find(words[i]) != -1 and 1 or 0) for doc in docs]        
        dfs.append(oneHot.count(1))
        #print('word',words[i],'df:',oneHot.count(1))
    print('calc df way2:', dfs)
    
    #calc df way3, 包含文辞的文档个数
    dfs, oneHots = [],[]
    for word in words:
        oneHots.append([(e.find(word) != -1 and 1 or 0) for e in docs])
    dfs.extend(e.count(1) for e in oneHots)
    print('calc oneHots way3:', np.array(oneHots))
    print('calc df way3:', dfs)
    
    #calc df way4, 包含词的文档个数
    dfs = []
    oneHots = [[doc.find(word) != -1 and 1 or 0 for doc in docs] for word in words]
    dfs.extend(e.count(1) for e in oneHots)
    print('calc oneHots way4:', np.array(oneHots))
    #dfs = np.reshape(dfs, (np.shape(dfs)[0],1)) #列向量1×n
    #print('calc df way4:', dfs)
    
    #calc idf, 计算每个词的idf(逆向文件频率inverse document frequency)
    #log10(N/(1+DF))
    N = np.shape(docs)[0]
    idfs = [(np.log10(N*1.0/(1+e))) for e in dfs]#f(e) = np.log10(N*1.0/(1+e))
    print('idfs:',np.array(idfs))
    
    #calc tf-idf,计算term frequency - inverse document frequency
    tfidfs = []
    for i in range(np.shape(docs)[0]):
        word_tfidf = np.multiply(tfs[i], idfs)
        tfidfs.append(word_tfidf)
        #print('word_tfidf:',word_tfidf)
    print('calc tfidfs:\n', np.array(tfidfs))
    
    print('==================result============================')
    print('\ndocs:\n', np.array(docs))
    
    print('\nwords:\n', np.array(words))
    
    print('\noneHots:\n', np.array(oneHots))
    
    print('\nCF:\n', np.array(cfs))
    
    print('\nTF:\n', np.array(tfs))
    
    print('\nDF:\n', np.array(dfs))
    
    print('\nIDF:\n', np.array(idfs))
    
    print('\nTF-IDF:\n', np.array(tfidfs))
    print('==============================================')
    return    
 
if __name__=='__main__':
    tfidf_alg()
    #sklearn_tfidf()
    