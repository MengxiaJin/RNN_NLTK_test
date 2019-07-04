from nltk import *
from nltk import word_tokenize

# 读取文本
text=open('origin_data/oridata.txt', "r", encoding='UTF-8').read()
#转列表
tokens = word_tokenize(text)
contents=[]
#删选有用信息
for token in tokens:
    if len(token)<20 or len(token)>100:
        continue
    contents.append('s'+token+'e')

#按长度排序
contents=sorted(contents,key=lambda x:len(x))
print("contents",contents[1])
words=[]
#频率排序
for content in  contents:
    words.extend(w for w in content)
fdist = FreqDist(words)
fdist = sorted(fdist.items(),key = lambda x:x[1],reverse = True)
#频率排序后的写入文件
word_id_dict={}
with open('vocab/poetry.vocab', 'w') as f:
    for id_l in fdist:
        f.write(id_l[0] + '\n')
        word_id_dict[id_l[0]]=fdist.index(id_l)
#向量化后的写入文件
id_list=[]
with open('processed_data/poetry.txt', 'w') as f:
    for content in contents:
        for word in content:
            if word in word_id_dict:
                id_list.append(word_id_dict[word])
                f.write(' '+str(word_id_dict[word]))
            else:
                id_list.append('无')
                f.write(' '+'无')
        f.write('\n')