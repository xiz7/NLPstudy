class DIC():
    def __init__(self,d,size):
        self.d = d
        self.size=size
#正向最大匹配算法
class MM(object):
        
    def cut(self,text,dic):
        i=0
        result = []
        text_length=len(text)
        while text_length>i:
            for size in range(dic.size+i,i,-1):
                p=text[i:size]
                if p in dic.d:
                    i = size-1
                    break
            i=i+1
            result.append(p+' / ')
        print(result)

text = '研究生命的起源'
dic = DIC(['研究','研究生','生命','命','的','起源'],3)
tokenizer=MM()

print(tokenizer.cut(text,dic))