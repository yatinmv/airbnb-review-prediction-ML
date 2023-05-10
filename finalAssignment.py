import numpy as np
import pandas as pd
from textblob import TextBlob
df = pd.read_csv('listings.csv')

df.head()

df.drop(
        [
            "listing_url",
            "scrape_id",
            "last_scraped",
            "source",
            "name",
            "description",
            "neighborhood_overview",
            "picture_url",
            "host_id",
            "host_url",
            "host_name",
            "host_location",
            "host_about",
            "host_thumbnail_url",
            "host_picture_url",
            "neighbourhood_group_cleansed",
            "bathrooms",
            "calendar_updated",
            "reviews_per_month",
            "license"
        ],
        axis=1,
        inplace=True,
    )

df = df.dropna()
df2 =  pd.read_csv('reviews.csv')
df2.dropna()
df2.head()


from langdetect import detect
def lang_detection(series):
    isEn = []
    for text in series:
        if len(text)<1:
            continue
        try:
            language = detect(text)
        except:
            language = "error"
            print("This row throws and error:", text)
            continue
        
        if language == 'en':
            isEn.append(text)
            
    return isEn


from nltk.corpus import wordnet as wn
import re


def synset_description(word):
    for synset in wn.synsets(word):
        print("%s : %s" % (synset.name(), synset.definition()))

def synonym_words_in(word, include_list):
    thesaurus = []
    for synset in wn.synsets(word):
        if synset.name() in include_list:
            thesaurus += synset.lemma_names()
    
    return list(set(thesaurus))

def synonym_words_ex(word, exclude_list):
    thesaurus = []
    for synset in wn.synsets(word):
        if synset.name() not in exclude_list:
            thesaurus += synset.lemma_names()
    
    return list(set(thesaurus))

def anton_list(syn_include_list):
    
    anton_list = []
    for synset in syn_include_list:
        for l in wn.synset(synset).lemmas():
            if l.antonyms():
                anton_list += l.antonyms()
    
    return anton_list

def synonyms_of_antonyms(anton_list):
    antonyms = []
    pattern = r"Lemma\('([a-z]+.[a-z].\d+)"
    
    for anton in anton_list:
        antonyms += wn.synset(re.findall(pattern, str(anton))[0]).lemma_names()
    
    return list(set(antonyms))

def communication_keyword_selection():
    synset_description('communication')
    synw1 = synonym_words_ex('communication', ['communication.n.03'])
    synset_description('message')
    synw2 = synonym_words_in('message', ['message.n.01','message.n.02 '])
 
    syn_include_list = ['communication.n.01','communication.n.02','message.n.01','message.n.02 ']

    antw = synonyms_of_antonyms(anton_list(syn_include_list))
    keywords = list(set(synw1  + synw2+ antw))
    
    return keywords


def cleanliness_keyword_selection():
    synset_description('cleanliness')
    synw1 = synonym_words_ex('cleanliness', '')
    synset_description('tidy')
    synw2 = synonym_words_in('tidy', ['tidy.v.01', 'tidy.a.01'])
    synset_description('clean')
    synw3 = synonym_words_in('clean', ['clean.v.01', 'houseclean.v.01', 'clean.v.05', 'clean.v.08', 
                                    'scavenge.v.04', 'clean.a.01'])


    # this is a list of synsets that are selected as useful synset in finding synonyms.
    syn_include_list = ['cleanliness.n.01', 'cleanliness.n.02', 'tidy.v.01', 'tidy.a.01', 
                    'clean.v.01', 'houseclean.v.01', 'clean.v.05', 'clean.v.08', 
                    'scavenge.v.04', 'clean.a.01']

    # a list of antonym synsets for above list.
    anton_list(syn_include_list)

    # all lemma words in upper antonym synsets.
    antw = synonyms_of_antonyms(anton_list(syn_include_list))
    cleanliness_synonyms = list(set(synw1 + synw2 + synw3 + antw))
    return cleanliness_synonyms

def value_keyword_selection():
    synset_description('worth')
    synw1 = synonym_words_ex('worth', ['worth.n.03','deserving.s.01'])
    synset_description('price')
    synw2 = synonym_words_in('price', ['price.n.02','price.n.03','price.n.04'])
    synset_description('value')
    synw3 = synonym_words_in('value', ['value.n.02', 'value.n.03', 'rate.v.03'])
    
    syn_include_list = ['price.n.02','price.n.03','price.n.04','value.n.02', 'value.n.03', 'rate.v.03','worth.n.01','worth.n.02','worth.s.02']
    antw = synonyms_of_antonyms(anton_list(syn_include_list))
    keywords = list(set(synw1  + synw2+ synw3 + antw))

    return keywords


def neighbourhood_keyword_selection():
    synset_description('location')
    synw1 = synonym_words_in('location', ['location.n.01'])
    synset_description('neighborhood')
    synw2 = synonym_words_ex('neighborhood', ['neighborhood.n.02'])

    # this is a list of synsets that are selected as useful synset in finding synonyms.
    syn_include_list = ['location.n.01', 'vicinity.n.01', 'region.n.04', 'neighborhood.n.04']

    # a list of antonym synsets for above list.
    anton_list(syn_include_list)

    # all lemma words in upper antonym synsets.
    antw = synonyms_of_antonyms(anton_list(syn_include_list))
    keywords = list(set(synw1 + synw2  + antw))
    return keywords

def checkin_keyword_selection():
    synset_description('check-in')
    synw1 = synonym_words_in('check-in','')
    
    # this is a list of synsets that are selected as useful synset in finding synonyms.
    syn_include_list = ['check-in.n.01']

    # a list of antonym synsets for above list.
    anton_list(syn_include_list)

    # all lemma words in upper antonym synsets.
    antw = synonyms_of_antonyms(anton_list(syn_include_list))
    keywords = list(set(synw1 + antw))
    return keywords

def accuracy_keyword_selection():
    synset_description('representation')
    synw1 = synonym_words_in('representation', ['representation.n.02'])
    synset_description('Information')
    synw2 = synonym_words_in('Information', ['information.n.01','information.n.02'])

#     # this is a list of synsets that are selected as useful synset in finding synonyms.
    syn_include_list = ['representation.n.02','information.n.01','information.n.02']
#     # all lemma words in upper antonym synsets.
    antw = synonyms_of_antonyms(anton_list(syn_include_list))
    keywords = list(set(synw1  + synw2+ antw))
    return keywords


import re
from pandas import Series, DataFrame
import pandas as pd
from numpy.random import randn
import numpy as np
import matplotlib.pyplot as plt
import os
import nltk
# use wordnet find the first level keywords
from nltk.corpus import wordnet as wn
import spacy

nlp = spacy.load('en_core_web_md')


def similar_sents(doc,threshold):
    similar_sents = []
    aspect_keywords = nlp(' '.join(keywords))
    
    for which_sen in range(len(list(doc.sents))):
        new_doc = list(doc.sents)[which_sen].text        
        sen_keywords = nlp(new_doc)
        
        if compute_score(sen_keywords, aspect_keywords, threshold) > 0:
            similar_sents.append(new_doc)
    
    return similar_sents

def compute_score(sen_keywords, aspect_keywords, threshold):
    count = 0
    for token1 in sen_keywords:
        for token2 in aspect_keywords:
            # print("\n token1:" , token1)
            # print("\n token2:" , token2)
            # print("\n Similarity: ", token1.similarity(token2))
            if token1.similarity(token2) >= threshold:
                count += 1 
                break
        if count>=1:
            break        
    return count

def clean_Comments(comments):
    result = []
    result =  [re.sub(r'[^\w\s]', '', str(x)).lower() for x in comments]
    return [re.sub(r'\d+', '', str(x)) for x in result]

def remove_automated_reviews(comments):
    
    filtered_comments = []

    for comment in comments:
        if 'automated posting' not in comment:
            filtered_comments.append(comment)
    return filtered_comments




# keywords = neighbourhood_keyword_selection()
# keywords = communication_keyword_selection()
# keywords = cleanliness_keyword_selection()
# keywords = checkin_keyword_selection()
# keywords = value_keyword_selection()
# keywords = accuracy_keyword_selection()

sentiments_mean = []
for i in df.iloc[:, 0]:
    reviews = df2.loc[df2['listing_id'] == i]
    comments = clean_Comments(reviews['comments'])
    comments = lang_detection(comments)
    comments = remove_automated_reviews(comments)
    print(comments)
# uncomment the below lines when finding the sentiment score for ratings other than the overall rating.

    # comments2 = ''.join(comments)
    # doc = nlp(comments2)
    # # print(doc.sents)
    # relavant_reviews = similar_sents(doc,0.7)
    # if len(relavant_reviews)==0:
    #     sentiments_mean.append(0)
    #     continue
    sentiments = []
    if len(comments)==0:
        sentiments_mean.append(0)
        continue
    for review in comments:
        blob = TextBlob(review)
        sentiments.append(blob.sentiment.polarity)
    sentiments_mean.append(np.mean(sentiments))

df['sentiment_score'] = sentiments_mean   
# df.to_csv('cleanliness_data.csv')
# df.to_csv('neighbourhood_data.csv')
# df.to_csv('communication_data.csv')
# df.to_csv('value_data.csv')
# df.to_csv('accuracy_data.csv')
df.to_csv('overallRating_data.csv')

