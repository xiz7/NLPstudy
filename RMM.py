class DIC():
    def __init__(self,d,size):
        self.d = d
        self.size=size
#正向最大匹配算法
class RMM(object):
        
    def cut(self,text,dic):
        
        result = []
        text_length=len(text)
        i=text_length-1
        while i>=0:
            for size in range(i-dic.size+1,i+1,1):
                p=text[size:i+1]
                print(p)
                if p in dic.d:
                    
                    i = size
                    break
            i=i-1
            result.append(p+' / ')
        print(result)

text = '研究生命的起源'
dic = DIC(['研究','研究生','生命','命','的','起源'],3)
tokenizer=RMM()

print(tokenizer.cut(text,dic))