#%% package
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import konlpy
from konlpy.tag import Okt
import re     
import gensim
from gensim import corpora
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import seaborn as sns
import nltk

#%% data 불러오기
row_data_before= pd.read_excel("C:/Users/user/Documents/GitHub/Kmong_project/Kmong_project6/data/2018-2019_medicalWear.xlsx",header = 0,index_col =None)
row_data_after= pd.read_excel("C:/Users/user/Documents/GitHub/Kmong_project/Kmong_project6/data/COVID19(2020-2021)_medicalWear.xlsx",header = 0,index_col =None)

row_data = pd.DataFrame()
row_data = pd.concat([row_data_before,row_data_after], axis=0,ignore_index=1)
row_data["input"] = row_data["제목"].astype(str) + " " +row_data["내용"].astype(str)


#%% 전처리 함수
from eunjeon import Mecab
from konlpy.tag import Mecab 
from tqdm import tqdm 
import re 
import pickle 
import csv
def clean_text(text):
    """ 한글, 영문, 숫자만 남기고 제거한다. :param text: :return: """ 
    text = text.replace(".", " ").strip() 
    text = text.replace("·", " ").strip() 
    pattern = '[^ ㄱ-ㅣ가-힣|0-9|a-zA-Z]+' 
    text = re.sub(pattern=pattern, repl='', string=text) 
    return text


def get_nouns(tokenizer, sentence):
    """ 단어의 길이가 2이상인 일반명사(NNG), 고유명사(NNP), 외국어(SL),동사(VV)만을 반환한다. :param tokenizer: :param sentence: :return: """ 
    tagged = tokenizer.pos(sentence) 
    nouns = [s for s, t in tagged if t in ['SL', 'NNG','VV'] and len(s) > 1] 
    return nouns

def tokenize(df):
    """ 한국어 tokenizer Mecab 패키지 사용 """ 
    tokenizer = Mecab()#(dicpath='C:/mecab/mecab-ko-dic')
    processed_data = [] 
    for sent in tqdm(df): 
        sentence = clean_text(sent.replace('\n', '').strip()) 
        processed_data.append(get_nouns(tokenizer, sentence)) 
    return processed_data

def apply_stop_words_covid(tokenized_text):
    """ 추출된 토픽에서 의미적으로 필요없는 단어 제거 """
    stop_words = pd.read_csv("C:/Users/user/Documents/GitHub/Kmong_project/Kmong_project6/stopwords.csv",header=0, names=["word"] ,encoding= "CP949")["word"].to_list()
    stop_words.extend(["죽음","담배","멸망","충북","주문","감사","죄수복","안녕","반티","서울",
                 "이날","공개","택시","검진","다음","가능","엄마","경찰","사랑","정신","순간","탈출",
                 "준비","정도","오후","소망","청주","국립","도주","일기","고양이","가슴","신고","생각",
                 "겨울","당국","놀이","카페","드라마","모습","자신","촬영","패션","이송","어린이",
                 "당시","관리","치매","사용","새벽","동안","침대","천안","출산","아버지","어머니","보이","할아버지","할머니"])
    result = []
    for tok_list in tokenized_text:
        tok_result =[]
        for tok in tok_list:
            if tok not in stop_words:
                tok_result.append(tok)
        result.append(tok_result)
    return result

  

from eunjeon import Mecab
tokenized_doc = tokenize(row_data["input"])
tokenized_doc = apply_stop_words_covid(tokenized_doc)
row_data["after preprocessing"] =tokenized_doc
'''
#tokenized_doc 저장
row_data.to_csv("C:/Users/user/Documents/GitHub/Kmong_project/Kmong_project6/data/after_covid19_data.csv",header = True, index = True, encoding='utf-8-sig')

import pickle
#covid
file_path = "C:/Users/user/Documents/GitHub/Kmong_project/Kmong_project6/data/tokenized_doc_version1(covid).pickle"
with open(file_path,"wb") as fw:
    pickle.dump(tokenized_doc, fw)
'''
#%% LDA 모델 실행
   
def make_topic_model(ldamodel,num_topic,num_word):
    
    topics = ldamodel.print_topics(num_topics= num_topic, num_words= num_word)
    df_topic = pd.DataFrame(topics)
    #단어 토큰화 작업 및 불필요 텍스트 제거
    tokenized_topic = [nltk.word_tokenize(doc.lower()) for doc in df_topic[1]]
    clean_topics= []
    for word in tokenized_topic:
        list_par = []
        for i in word:
            text = re.sub('[^ㄱ-ㅎㅏ-ㅣ가-힣|a-zA-Z]+',' ',i).strip() #한글,제외 다 제거.
            if(text != ''): # 공백 제거
                list_par.append(text)
        clean_topics.append(list_par)

    #DataFrame으로 틀만들어서 넣기    
    clean_topics= pd.DataFrame(clean_topics)
    df_topics =clean_topics.transpose()
    df_topics.columns = ['Topic'+str(i) for i in range(1,num_topic+1)]
                                                   
    # LDA modeling 결과 csv 파일 저장
    modeling_name = file_path + now+'Topic=' + str(num_topic)+ '_modeling.csv'
    df_topics.to_csv(modeling_name, index=True,encoding = "CP949")
    print("make topic model complete!")
    return df_topics
 

# Topic table 적용
def make_topic_table(ldamodel, corpus, num_topic):
    topictable = pd.DataFrame()

    # 몇 번째 문서인지를 의미하는 문서 번호와 해당 문서의 토픽 비중을 한 줄씩 꺼내온다.
    for i, topic_list in enumerate(ldamodel[corpus]):
        doc = topic_list[0] if ldamodel.per_word_topics else topic_list            
        doc = sorted(doc, key=lambda x: (x[1]), reverse=True)
        # 각 문서에 대해서 비중이 높은 토픽순으로 토픽을 정렬한다.
        # EX) 정렬 전 0번 문서 : (2번 토픽, 48.5%), (8번 토픽, 25%), (10번 토픽, 5%), (12번 토픽, 21.5%), 
        # Ex) 정렬 후 0번 문서 : (2번 토픽, 48.5%), (8번 토픽, 25%), (12번 토픽, 21.5%), (10번 토픽, 5%)
        # 48 > 25 > 21 > 5 순으로 정렬이 된 것.

        # 모든 문서에 대해서 각각 아래를 수행
        for j, (topic_num, prop_topic) in enumerate(doc): #  몇 번 토픽인지와 비중을 나눠서 저장한다.
            if j == 0:  # 정렬을 한 상태이므로 가장 앞에 있는 것이 가장 비중이 높은 토픽
                topictable = topictable.append(pd.Series([int(topic_num), round(prop_topic,10), topic_list]), ignore_index=True)
                # 가장 비중이 높은 토픽과, 가장 비중이 높은 토픽의 비중과, 전체 토픽의 비중을 저장한다.
            else:
                break
    topictable = topictable.reset_index() # 문서 번호을 의미하는 열(column)로 사용하기 위해서 인덱스 열을 하나 더 만든다.
    topictable.columns = ['문서 번호', '가장 비중이 높은 토픽', '가장 높은 토픽의 비중', '각 토픽의 비중']
           
    #LDA table 결과 csv 저장
    table_name = file_path + now + 'Topic=' + str(num_topic)+ '_table.csv'
    topictable.to_csv(table_name, index=True, encoding = "utf-8-sig")
    print("make topic table complete!")
    return topictable

# 문서별 토픽 유사도+ visualizer
def make_topic_simliarity(ladmodel,corpus,num_topic, num_cluster):
    simliarity_vetor=[]
    for i in range(len(corpus)):
        r=[]
        for w in ldamodel.get_document_topics(corpus[i], minimum_probability=0):
            r.append(w[1])
        simliarity_vetor.append(r)
    E= pd.DataFrame(simliarity_vetor)
    E.to_csv(file_path +now+ 'Topic='+str(num_topic)+'_simliarity.csv', header= ["topic"+str(i) for i in range(1, num_topic+1)])
    print("make topic simliarity complete!")
    
    kmeans = KMeans(n_clusters= num_cluster).fit(simliarity_vetor)
    clusters = kmeans.labels_
    TSNE_vetor = TSNE(n_components=2).fit_transform(simliarity_vetor)# component = 차원
    Q = pd.DataFrame(TSNE_vetor) # dataframe으로 변경하여 K-means cluster lavel 열 추가
    Q["clusters"] = clusters #lavel 추가
    fig, ax = plt.subplots(figsize=(12,8))
    sns.scatterplot(data = Q, x=0, y=1, hue= clusters, palette='deep')
    plt.show()
    print("visualizer complete!")
    

#%% LDA 구현 전 dictionary 구축

num_topic = 100
dictionary = corpora.Dictionary(tokenized_doc) # tokenized 데이터를 통해 dictionary로 변환
dictionary.filter_extremes(no_below=15, no_above =0.1) #5회 이하로 등장한 단어는 삭제 15,0.1
corpus = [dictionary.doc2bow(text) for text in tokenized_doc] # 코퍼스 구성
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = num_topic, id2word=dictionary, iterations= 300, passes =10, random_state = 1004 ) #300,10
len(dictionary)

# LDA 구현 
now = '220607_' # 오늘날짜
file_path = "C:/Users/user/Documents/GitHub/Kmong_project/Kmong_project6/output/" # 저장 경로
# topic model
topic_model = make_topic_model(ldamodel, num_topic=num_topic, num_word=100)
# topic table
topic_table = make_topic_table(ldamodel, corpus, num_topic =num_topic)   


#%% 통계적 검정
Before_covid = topic_table[:4924]
After_covid = topic_table[4924:]
Before_covid = Before_covid["가장 비중이 높은 토픽"].replace(0,"topic1").replace(1,"topic2").replace(2,"topic3").replace(3,"topic4")
After_covid = After_covid["가장 비중이 높은 토픽"].replace(0,"topic1").replace(1,"topic2").replace(2,"topic3").replace(3,"topic4")

Before_covid["가장 비중이 높은 토픽"].value_counts()/len(Before_covid)#모비율 topic2 = 0.287
After_covid["가장 비중이 높은 토픽"].value_counts()/len(After_covid) #모비율 topic2 = 0.334

#복원추출 시뮬레이터
B_rate_list =pd.DataFrame()
A_rate_list =pd.DataFrame() 
for col in ["topic1","topic2","topic3","topic4"]:
    B_list= []
    A_list = []
    for i in range(10000):
        B_sample = Before_covid.sample(n=1000, replace=True).value_counts(normalize = True)[col]
        A_sample = After_covid.sample(n=1000, replace=True).value_counts(normalize = True)[col]
        B_list.append(B_sample)
        A_list.append(A_sample-B_sample)
    B_rate_list[col] = B_list
    A_rate_list[col] = A_list
 
    

#평균
def mean(inp):
    result = 0
    len_inp = len(inp)    
    for i in inp:
        result += i
    result = result / len_inp
    return result

#분산
def var(inp):
    result = 0
    len_inp = len(inp)
    for i in inp:
        result += (i - mean(inp)) ** 2
    result = result / len_inp
    return result

#제곱근
def sqrt(inp):
    result = inp/2
    for i in range(30):
        result = (result + (inp / result)) / 2
    return result

#표준편차
def std(inp):
    result = sqrt(var(inp))
    return result

#%%사분위수로 신뢰구간 구하기
print(pd.DataFrame(A_rate_list).quantile(q=0.5))
print(pd.DataFrame(B_rate_list).quantile(q=0.5))
print(pd.DataFrame(A_rate_list).quantile(q=0.05))
print(pd.DataFrame(A_rate_list).quantile(q=0.95))

#%% 토픽 비율 그래프

# 도화지
plt.style.use("default")
fig, ax = plt.subplots()
fig.set_size_inches(12, 9)

# 평균
ax.plot(["Topic1","Topic2","Topic3","Topic4"], pd.DataFrame(A_rate_list).quantile(q=0.5),linestyle = "-",color = "black",linewidth = 4)
ax.plot(["Topic1","Topic2","Topic3","Topic4"], pd.DataFrame(B_rate_list).quantile(q=0.5), linestyle = "--",color = "black",linewidth = 4)

# x축, y축 폰트 사이즈
ax.tick_params(axis = 'x', labelsize = 10)
ax.tick_params(axis = 'y', labelsize = 10)

#legend
ax.set_xlabel("Name of topics",fontsize = 15)
ax.set_ylabel("Average topic ratio",fontsize = 15)
plt.legend(['After_COVID19', 'Before_COVID19'],fontsize =15)
plt.title("Rate of topics",fontsize = 20)

plt.show()
#%% 신뢰구간 그래프
# 도화지
plt.style.use("default")
fig, ax = plt.subplots()
fig.set_size_inches(12, 9)

# 신뢰구간
plt.errorbar(["Topic1","Topic2","Topic3","Topic4"], pd.DataFrame(A_rate_list).quantile(q=0.5), yerr=(pd.DataFrame(A_rate_list).quantile(q=0.975)-pd.DataFrame(A_rate_list).quantile(q=0.025))/2, 
             ecolor = "black",capsize =5,marker ="o",color = "black",linewidth = 4,alpha = 0.2)

#plt.errorbar(["Topic1","Topic2","Topic3","Topic4"], pd.DataFrame(B_rate_list).quantile(q=0.45), yerr=(pd.DataFrame(B_rate_list).quantile(q=0.975)-pd.DataFrame(B_rate_list).quantile(q=0.025))/2,

ax.plot(["Topic1","Topic2","Topic3","Topic4"], pd.DataFrame(A_rate_list).quantile(q=0.5),linestyle = "-",color = "black",linewidth = 4)
ax.plot(["Topic1","Topic2","Topic3","Topic4"], [0,0,0,0],linestyle = ":",color = "black",linewidth = 4)
# x축, y축 폰트 사이즈
ax.tick_params(axis = 'x', labelsize = 10)
ax.tick_params(axis = 'y', labelsize = 10)

#legend
ax.set_xlabel("Name of topics",fontsize = 15)
ax.set_ylabel("topic differences ratio",fontsize = 15)
#plt.legend(['After_COVID19 - Before_COVID19'],fontsize =15)
plt.title("Rate of topic differences(After COVID19 - Before COVID19)",fontsize = 20)

plt.show()

