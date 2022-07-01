#%% package
from tqdm import tqdm
from gensim.models.ldamodel import LdaModel 
from gensim.models.callbacks import CoherenceMetric 
from gensim import corpora 
from gensim.models.callbacks import PerplexityMetric 
import logging 
import pickle
from gensim.models.coherencemodel import CoherenceModel 
import matplotlib.pyplot as plt 
import numpy as np
#%% 평가지
# coherence 함수

def coherence_optimal_number_of_topics(dictionary, corpus, processed_data): 
    limit = 20; #토픽 마지막갯수
    start = 2; #토픽 시작갯수
    step = 2; 
    coherence_values = []
    for num_topics in range(start, limit, step):
        lda_model = LdaModel(corpus, id2word=dictionary, num_topics=num_topics, passes=10, iterations= 300, random_state=1004)
        cm = CoherenceModel(model= lda_model, corpus = corpus, coherence= 'u_mass')
        coherence_values.append(cm.get_coherence())
           
    x = range(start, limit, step) 
    plt.plot(x, coherence_values) 
    plt.xlabel("Num Topics") 
    plt.ylabel("Coherence")
    plt.show() 
    
#perplexity 함수

def perplexity_optimal_number_of_topics(dictionary, corpus, processed_data): 
    limit = 20; #토픽 마지막갯수
    start = 2; #토픽 시작갯수
    step = 2; 
    perplexity_values = []
    for num_topics in range(start, limit, step):
        lda_model = LdaModel(corpus, id2word=dictionary, num_topics=num_topics, passes=10, iterations= 300, random_state = 1004)
        perplexity_values.append(lda_model.log_perplexity(corpus))
        
    x = range(start, limit, step) 
    plt.plot(x, perplexity_values) 
    plt.xlabel("Num Topics") 
    plt.ylabel("Perplexity")
    plt.show() 


#%% 실행

#preprocessing 완료된 document pickle 파일 열기
import pickle
file_path = "C:/Users/user/Documents/GitHub/Kmong_project/Kmong_project6/data/tokenized_doc_version1(covid).pickle"
file_path = "C:/Users/user/Documents/GitHub/Kmong_project/Kmong_project6/data/tokenized_doc_version2(mers).pickle"
          
with open(file_path,"rb") as fr:
          tokenized_doc = pickle.load(fr)
          
 # 출현빈도가 적거나 자주 등장하는 단어는 제거 
dictionary = corpora.Dictionary(tokenized_doc)
dictionary.filter_extremes(no_below=20, no_above=0.1)
corpus = [dictionary.doc2bow(text) for text in tokenized_doc]
print('Number of unique tokens: %d' % len(dictionary)) 
print('Number of documents: %d' % len(corpus))

# 최적의 토픽 수 찾기 
coherence_optimal_number_of_topics(dictionary, corpus, tokenized_doc)
perplexity_optimal_number_of_topics(dictionary, corpus, tokenized_doc)

#최적의 pass 수 찾기(num_topic 고정)

coherences=[]
perplexities=[]
passes=[]

x = range(10,110,10)
plt.plot(x,coherences)
plt.xlabel("Pass") 
plt.ylabel("Coherence")
plt.show() 

x = range(10,110,10)
plt.plot(x,perplexities)
plt.xlabel("Pass") 
plt.ylabel("Perplexity")
plt.show() 

for p in range(10,110,10):
    passes.append(p) 
    lda = LdaModel(corpus, id2word=dictionary, num_topics=30, iterations=400, passes=p)
    cm = CoherenceModel(model=lda, corpus=corpus, coherence='u_mass')
    coherence = cm.get_coherence()
    print("Cpherence",coherence)
    coherences.append(coherence)
    print('Perplexity: ', lda.log_perplexity(corpus),'\n\n')
    perplexities.append(lda.log_perplexity(corpus))
