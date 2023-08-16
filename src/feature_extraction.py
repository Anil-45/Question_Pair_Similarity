"""Feature extraction functions."""

import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra
from scipy.spatial.distance import euclidean, minkowski, braycurtis
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import distance
import fuzzywuzzy as fuzz
import spacy
from tqdm import tqdm
from preprocessing import text_pre_processor
from utils import save_dict, load_dict, reduce_mem_usage

def word_feat(row):
    """Word based features.
    
    Args:
        row : DataFrame row

    Returns:
        bool: First word equal
        bool: Last word equal
        int : Number of unique common words in q1 and q2
        int : Number of unique words in both q1 and q2
    """
    w1 = row.question1.lower().split(" ")
    w2 = row.question2.lower().split(" ")
    
    # checking first word and last word
    r1, r2 = (int(w1[0] == w2[0]), int(w1[-1] == w2[-1]))
    
    w1 = set(w1)
    w2 = set(w2)   
    
    # common words, total words
    r3, r4 = (len(w1.intersection(w2)), len(w1)+len(w2))
    return (r1, r2, r3, r4)


def get_basic_features(data):
    """Get basic features.
    
    Args:
        data (pd.DataFrame): input
    
    Returns:
        pd.DataFrame: DataFrame with basic features
    """
    df = data.copy()
    
    df['question1'] = df['question1'].apply(text_pre_processor)
    df['question2'] = df['question2'].apply(text_pre_processor)
    
    df['q1_len'] = df['question1'].apply(lambda x: len(str(x)))
    df['q2_len'] = df['question2'].apply(lambda x: len(str(x)))
    df['diff_len'] = abs(df['q1_len'] - df['q2_len'])
    df['avg_len'] = (df['q1_len'] + df['q2_len'])/2
    
    df['q1_n_words'] = df['question1'].apply(lambda x: len(str(x).split(" ")))
    df['q2_n_words'] = df['question2'].apply(lambda x: len(str(x).split(" ")))
    df['diff_word'] = abs(df['q1_n_words'] - df['q2_n_words'])
    df['avg_words'] = (df['q1_n_words'] + df['q2_n_words'])/2
    
    #  word_common = (Number of common unique words in Question 1 and Question 2)
    #  word_Total = (Total num of words in Question 1 + Total num of words in Question 2)
    #  word_share = (word_common)/(word_Total)
    
    df[['first_same', 'last_same', 'word_common', 'word_total']] = df.apply(lambda row: pd.Series(word_feat(row)), axis=1)
    df['word_share'] = round(df['word_common']/df['word_total'], 2)
    
    return df

def get_stopword_features(q1, q2):
    """Stopword features.
    
    Args:
        q1 (String): Question 1
        q2 (String): Question 2
    
    Returns:
        list: Stopword features
    """
    SAFE_DIV = 0.0001 
    STOP_WORDS = nltk.corpus.stopwords.words("english")

    features = [0.0]*4
    
    # Converting the Sentence into Tokens
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return features
    
    # Get the non-stopwords in Questions
    q1_non_stop_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_non_stop_words = set([word for word in q2_tokens if word not in STOP_WORDS])
    
    # Get the stopwords in Questions
    q1_stop_words = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stop_words = set([word for word in q2_tokens if word in STOP_WORDS])
    
    # Get the common non-stopwords from Question pair
    common_non_stop_words_count = len(q1_non_stop_words.intersection(q2_non_stop_words))
    
    # Get the common stopwords from Question pair
    common_stop_words_count = len(q1_stop_words.intersection(q2_stop_words))
    
    # cnsc_min : (common non stop words count) / min(len_q1_non_stop_words, len_q2_non_stop_words) 
    features[0] = common_non_stop_words_count / (min(len(q1_non_stop_words), len(q2_non_stop_words)) + SAFE_DIV)
    # cnsc_max : (common non stop words count) / max(len_q1_non_stop_words, len_q2_non_stop_words) 
    features[1] = common_non_stop_words_count / (max(len(q1_non_stop_words), len(q2_non_stop_words)) + SAFE_DIV)
    # csc_min : (common stop words count) / min(len_q1_stop_words, len_q2_stop_words) 
    features[2] = common_stop_words_count / (min(len(q1_stop_words), len(q2_stop_words)) + SAFE_DIV)
    # csc_max : (common stop words count) / min(len_q1_stop_words, len_q2_stop_words)
    features[3] = common_stop_words_count / (max(len(q1_stop_words), len(q2_stop_words)) + SAFE_DIV)
    
    return features


def get_longest_substr_ratio(a, b):
    """LCS.

    Longest_substr_ratio = len(longest_common_substring)/min(len(str1), len(str2))
    
    Args:
        a (String): First string    
        b (String): Second string

    Returns:
        float: Longest_substr_ratio
    """
    longest_substrings = list(distance.lcsubstrings(a, b))
    if len(longest_substrings) == 0:
        return 0
    else:
        return len(longest_substrings[0]) / min(len(a), len(b))
    
def get_advanced_features(data):
    """Get advanced features.

    stopword_features
    fuzzy_features: http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/

    Args:
        data (pd.DataFrame): data

    Returns:
        pd.DataFrame: dataframe with advanced features
    """
    df = data.copy()
    
    df[['cnsc_min', 'cnsc_max', 'csc_min', 'csc_max']] = df.apply(
        lambda row: pd.Series(get_stopword_features(row['question1'],
                                                    row['question2'])), axis=1)
    
    df["token_set_ratio"]    = df.apply(lambda x: fuzz.token_set_ratio(
                                        x["question1"], x["question2"]), axis=1)
    df["token_sort_ratio"]   = df.apply(lambda x: fuzz.token_sort_ratio(
                                        x["question1"], x["question2"]), axis=1)
    df["fuzz_Qratio"]        = df.apply(lambda x: fuzz.QRatio(
                                        x["question1"], x["question2"]), axis=1)
    df["fuzz_partial_ratio"] = df.apply(lambda x: fuzz.partial_ratio(
                                        x["question1"], x["question2"]), axis=1)
    
    df["longest_substr_ratio"]  = df.apply(lambda x: get_longest_substr_ratio(
                                        x["question1"], x["question2"]), axis=1)
    return df


def get_vectors(data, word2idf):
    """Get vectors.

    Args:
        data (pd.DataFrame): data
        word2idf (dict): idf dict of of tfidf
    """
    df = data.copy()
    
    nlp = spacy.load('en_core_web_sm')

    for col in ['question1', 'question2']:
        vectors = []

        for doc in tqdm(nlp.pipe(df[col])):
            if len(doc) != 0:
                mean_vec = np.zeros(len(doc[0].vector))
            else:
                mean_vec = np.zeros(96)
                
            tf_idf_sum = 0
            for word in doc:
                vec = word.vector
                try:
                    tf_idf = (list(doc).count(word)/len(doc))*word2idf[str(word)]
                except:
                    tf_idf = 0
                mean_vec += vec * tf_idf
                tf_idf_sum += tf_idf
            if tf_idf_sum != 0: 
                mean_vec /= tf_idf_sum
            vectors.append(mean_vec)

        temp_df = pd.DataFrame(vectors)
        temp_df.columns = [f"feat_{col}_{i}" for i in range(len(vectors[0]))]
        temp_df['id'] = df['id']
        df = df.merge(temp_df, on='id', how='left')
   
    return df


def get_distance_feat(data):
    """Get distance features.
    
    Args:
        data (pd.DataFrame): data
    
    Returns:
        pd.DataFrame: DataFrame with distance features
    """
    df = data.copy()
    
    question1_vectors = df[[col for col in df.columns
                            if "feat_question1" in col]].values
    question2_vectors = df[[col for col in df.columns
                            if "feat_question2" in col]].values
    
    df['cosine_distance'] = [cosine(x, y) for (x, y) in 
                             zip(np.nan_to_num(question1_vectors),
                                 np.nan_to_num(question2_vectors))]

    df['cityblock_distance'] = [cityblock(x, y) for (x, y) in 
                                zip(np.nan_to_num(question1_vectors),
                                    np.nan_to_num(question2_vectors))]

    df['jaccard_distance'] = [jaccard(x, y) for (x, y) in 
                              zip(np.nan_to_num(question1_vectors),
                                 np.nan_to_num(question2_vectors))]

    df['canberra_distance'] = [canberra(x, y) for (x, y) in
                               zip(np.nan_to_num(question1_vectors),
                               np.nan_to_num(question2_vectors))]

    df['euclidean_distance'] = [euclidean(x, y) for (x, y) in
                                zip(np.nan_to_num(question1_vectors),
                                np.nan_to_num(question2_vectors))]

    df['minkowski_distance'] = [minkowski(x, y, 3)
                                for (x, y) in zip(np.nan_to_num(question1_vectors),
                                np.nan_to_num(question2_vectors))]

    df['braycurtis_distance'] = [braycurtis(x, y) for (x, y) in
                                 zip(np.nan_to_num(question1_vectors),
                                     np.nan_to_num(question2_vectors))]

    df['skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
    df['skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
    df['kur_q1vec']  = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
    df['kur_q2vec']  = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]
    return df

def generate_features(data, is_train=True):
    """Generate all features

    Args:
        data (pd.DataFrame): data
        is_train (bool, optional): is_train. Defaults to True.
        
    Returns:
        pd.DataFrame: DataFrame with only features.
    """
    df = data.copy()
    
    df = get_basic_features(df)
    df = get_advanced_features(df)
    
    if is_train:
        questions = df['question1'].to_list() + df['question2'].to_list()
        tfidf = TfidfVectorizer()
        tfidf.fit(questions)
        word2idf = dict(zip(tfidf.get_feature_names_out(),  tfidf.idf_))
        save_dict(word2idf, '../models/word2idf.pkl')
    else:
        assert os.path.exists('../models/word2idf.pkl'), 'word2idf.pkl missing'
        word2idf = load_dict('../models/word2idf.pkl')
    
    df = get_vectors(df, word2idf)
    
    if is_train:
        df.drop(columns=['id', 'qid1', 'qid2', 'question1', 'question2'],
                inplace=True)
        df.dropna(axis=0, inplace=True)
        
    df = get_distance_feat(df)
    df = reduce_mem_usage(df)
    return df
