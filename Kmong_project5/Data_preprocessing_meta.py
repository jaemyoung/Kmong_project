#한국어 LDA 전처리
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
###Data 전처리
pip install eunjeon

# data 불러오기
row_data_before = pd.read_excel("C:/Users/user/Documents/GitHub/Kmong_project/Kmong_project6/data/코로나전시기.xlsx",header = None,index_col =None,names = ["text"])
row_data_after= pd.read_excel("C:/Users/user/Documents/GitHub/Kmong_project/Kmong_project6/data/코로나시기.xlsx",header = None,index_col =None,names = ["text"])

row_data = pd.DataFrame()
row_data = pd.concat([row_data_before,row_data_after], axis=0,ignore_index=1)


from eunjeon import Mecab
m = Mecab()

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
    """ 단어의 길이가 2이상인 일반명사(NNG), 고유명사(NNP), 외국어(SL)만을 반환한다. :param tokenizer: :param sentence: :return: """ 
    tagged = tokenizer.pos(sentence) 
    nouns = [s for s, t in tagged if t in ['SL', 'NNG', 'NNP'] and len(s) > 1] 
    return nouns

def tokenize(df): 
    tokenizer = Mecab()#(dicpath='C:/mecab/mecab-ko-dic')
    processed_data = [] 
    for sent in tqdm(df['text']): 
        sentence = clean_text(sent.replace('\n', '').strip()) 
        processed_data.append(get_nouns(tokenizer, sentence)) 
    return processed_data



def apply_stop_words(tokenized_text):
    stop_words.extend(["명","후","실","진","비","복","부","좀","것"]) # 의미없는 단어 제거
    result = []
    for tok_list in tokenized_text:
        tok_result =[]
        for tok in tok_list:
            if tok not in stop_words:
                tok_result.append(tok)
        result.append(tok_result)
    return result  


tokenized_doc = tokenize(row_data)

tokenized_doc = apply_stop_words(tokenized_doc)


###LDA 모델 실행

# LDA 구현 
num_topic = 10

dictionary = corpora.Dictionary(tokenized_doc) # tokenized 데이터를 통해 dictionary로 변환
dictionary.filter_extremes(no_below=10, no_above =0.05) #20회 이하로 등장한 단어는 삭제
corpus = [dictionary.doc2bow(text) for text in tokenized_doc] # 코퍼스 구성
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = num_topic, id2word=dictionary, iterations= 100, passes = 10, random_state = 1004 ) 
'''
def make_topic_table(document_text, ldamodel, num_topic, num_word):
    topics = ldamodel.print_topics(num_topics= num_topic, num_words= num_word)
    df_topic = pd.DataFrame(topics)
    #topic 단어들 전처리
    tokenized_topic = [nltk.word_tokenize(doc.lower()) for doc in df_topic[1]]
    clean_topics= []
    for word in tokenized_topic:
        list_par = []
        for i in word:
            text = re.sub('[^ㄱ-ㅎㅏ-ㅣ가-힣]',' ',i).strip() # 한글제외 다 제거.
            if(text != ''): # 영어,숫자 및 공백 제거.
                list_par.append(text)
        clean_topics.append(list_par)
    
    df_topic["Keywords"] =clean_topics

    topictable = pd.DataFrame()
    for i, topic_list in enumerate(ldamodel[corpus]):
        doc = topic_list[0] if ldamodel.per_word_topics else topic_list            
        doc = sorted(doc, key=lambda x: (x[1]), reverse=True)

        # 모든 문서에 대해서 각각 아래를 수행
        for j, (topic_num, prop_topic) in enumerate(doc): #  몇 번 토픽인지와 비중을 나눠서 저장한다.
               if j == 0:  # 정렬을 한 상태이므로 가장 앞에 있는 것이 가장 비중이 높은 토픽
                   topictable = topictable.append(pd.Series([int(topic_num), round(prop_topic,10), topic_list]), ignore_index=True)
                # 가장 비중이 높은 토픽과, 가장 비중이 높은 토픽의 비중과, 전체 토픽의 비중을 저장한다.
               else:
                   break


    #전처리
    result_table = topictable.drop([2],axis=1) #원래있던 학률*topic 제거
    result_table = result_table.reset_index() # 문서 번호을 의미하는 열(column)로 사용하기 위해서 인덱스 열을 하나 더 만든다.
    result_table.columns = ['Document_No', 'Dominant_topic', 'Topic_Perc_Contrib'] #인덱스열 name 지정
    table = result_table.join(df_topic["Keywords"], on ="Dominant_topic") #topic에 따라 keywords 할당
    table["Text"] = document_text
    
    return table
'''
    

def make_topic_model(ldamodel,num_topic,num_word):
    
    topics = ldamodel.print_topics(num_topics= num_topic, num_words= num_word)
    df_topic = pd.DataFrame(topics)
    #단어 토큰화 작업 및 불필요 텍스트 제거
    tokenized_topic = [nltk.word_tokenize(doc.lower()) for doc in df_topic[1]]
    clean_topics= []
    for word in tokenized_topic:
        list_par = []
        for i in word:
            text = re.sub('[^ㄱ-ㅎㅏ-ㅣ가-힣]',' ',i).strip() #한글제외 다 제거.
            if(text != ''): # 영어,숫자 및 공백 제거.
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
    


###########################################################################
# 적용(LDA 구현했을 때 파라미터 동일하게 해야됨)
now = '220413_' # 오늘날짜
file_path = "C:/Users/user/Documents/GitHub/Kmong_project/Kmong_project6/output/"

topic_model = make_topic_model(ldamodel, num_topic=num_topic, num_word=20)
topic_table = make_topic_table(ldamodel, corpus, num_topic =num_topic)   
topic_table.columns


test = topic_table
test1 = test[:len(row_data_before)]
test2 = test[len(row_data_before):]
Before_covid = pd.DataFrame(data = test1["가장 비중이 높은 토픽"].value_counts()/len(test1))
After_covid = pd.DataFrame(data =test2["가장 비중이 높은 토픽"].value_counts()/len(test2))
Before_covid.rename(columns = {"가장 비중이 높은 토픽": "토픽의 비율"},inplace =True)
After_covid.rename(columns = {"가장 비중이 높은 토픽": "토픽의 비율"},inplace =True)
Before_covid["토픽"] = (Before_covid.index)+1
After_covid["토픽"] = (Before_covid.index)+1
Before_covid = Before_covid.reset_index(drop=True)
After_covid = After_covid.reset_index(drop=True)

#시각화
from matplotlib import pyplot as plt
x_values = ["Topic1","Topic2","Topic3","Topic4","Topic5","Topic6","Topic7","Topic8","Topic9","Topic10" ]

plt.plot(x_values, Before_covid.sort_values(by=["토픽"])["토픽의 비율"])
plt.plot(x_values, After_covid.sort_values(by=["토픽"])["토픽의 비율"])
plt.legend(['Before_covid', 'After_covid'])
plt.title("Rate of topics")
plt.show()

