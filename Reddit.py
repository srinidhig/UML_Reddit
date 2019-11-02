# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from pandas import DataFrame
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA

title = DataFrame.from_csv("Reddit_Title.tsv", sep="\t")

title = title.reset_index()

### DATA PREPARATION
target_sr = title[~title.TARGET_SUBREDDIT.isin(title.SOURCE_SUBREDDIT)]
target_sr['SOURCE_SUBREDDIT'] = target_sr['TARGET_SUBREDDIT']
target_sr['TARGET_SUBREDDIT'] = '-'
title = title.append(target_sr)
del target_sr

properties_df = title.PROPERTIES.str.split(',', expand = True)
properties_df.columns = ['Num_char',
'Num_char_nospace',
'Frac_alphabet',
'Frac_digits',
'Frac_upper',
'Frac_white',
'Frac_special',
'Num_words',
'Num_unique_words',
'Num_longwords',
'Avg_wordlength',
'Num_unique_stopwords',
'Frac_stopwords',
'Num_sentences',
'Num_longsentences',
'Avg_numchar_sentence',
'Avg_numchar_words',
'Readability_index',
'Pos_sentiment',
'Neg_sentiment',
'Compound_sentiment',
'LIWC_Funct',
'LIWC_Pronoun',
'LIWC_Ppron',
'LIWC_I',
'LIWC_We',
'LIWC_You',
'LIWC_SheHe',
'LIWC_They',
'LIWC_Ipron',
'LIWC_Article',
'LIWC_Verbs',
'LIWC_AuxVb',
'LIWC_Past',
'LIWC_Present',
'LIWC_Future',
'LIWC_Adverbs',
'LIWC_Prep',
'LIWC_Conj',
'LIWC_Negate',
'LIWC_Quant',
'LIWC_Numbers',
'LIWC_Swear',
'LIWC_Social',
'LIWC_Family',
'LIWC_Friends',
'LIWC_Humans',
'LIWC_Affect',
'LIWC_Posemo',
'LIWC_Negemo',
'LIWC_Anx',
'LIWC_Anger',
'LIWC_Sad',
'LIWC_CogMech',
'LIWC_Insight',
'LIWC_Cause',
'LIWC_Discrep',
'LIWC_Tentat',
'LIWC_Certain',
'LIWC_Inhib',
'LIWC_Incl',
'LIWC_Excl',
'LIWC_Percept',
'LIWC_See',
'LIWC_Hear',
'LIWC_Feel',
'LIWC_Bio',
'LIWC_Body',
'LIWC_Health',
'LIWC_Sexual',
'LIWC_Ingest',
'LIWC_Relativ',
'LIWC_Motion',
'LIWC_Space',
'LIWC_Time',
'LIWC_Work',
'LIWC_Achiev',
'LIWC_Leisure',
'LIWC_Home',
'LIWC_Money',
'LIWC_Relig',
'LIWC_Death',
'LIWC_Assent',
'LIWC_Dissent',
'LIWC_Nonflu',
'LIWC_Filler']

title = pd.concat([title.drop('PROPERTIES', axis = 1), properties_df], axis = 1)

numerical_cols = properties_df.columns.tolist() + ['LINK_SENTIMENT']
for col in numerical_cols:
    title[col] = title[col].astype(float)

title['TIMESTAMP'] = pd.to_datetime(title['TIMESTAMP'])

del properties_df

title['POST_ID'] = title['POST_ID'].str[:6]
title['YEAR'] = title.TIMESTAMP.astype(str).str[:4].astype(int)
title['MONTH'] = title.TIMESTAMP.astype(str).str[5:7].astype(int)
title['DAY'] = title.TIMESTAMP.astype(str).str[8:10].astype(int)
title['DAYOFWEEK'] = title.TIMESTAMP.dt.dayofweek.astype(int)
title['HOUR'] = title.TIMESTAMP.astype(str).str[11:13].astype(int)

### Dataset Summaries
summary = title.describe().T
summary_noconflicts = title[title.LINK_SENTIMENT == 1].describe().T
summary_conflicts = title[title.LINK_SENTIMENT != 1].describe().T

### Variable Standardization



### Initial clusters
idvs = ['LINK_SENTIMENT','Num_words','Frac_alphabet','Frac_digits','Frac_special','Avg_wordlength',
        'Readability_index','Compound_sentiment','LIWC_Funct','LIWC_Negate','LIWC_Swear','LIWC_Social',
        'LIWC_Affect','LIWC_CogMech','LIWC_Percept','LIWC_Bio','LIWC_Relativ','LIWC_Money','DAYOFWEEK','HOUR']

kmeans = KMeans(n_clusters=10, random_state=0, n_jobs=-1).fit(title[idvs])

title['Initial_Clusters'] = kmeans.labels_

cluster_characteristics = pd.DataFrame(kmeans.cluster_centers_)
cluster_characteristics = cluster_characteristics.join(title.Initial_Clusters.value_counts())
cluster_characteristics.columns = idvs + ['Row_Count']

### Clusters on data with features from PCA
pca = PCA(n_components=2, random_state=0).fit(title[idvs])
pca_components = pd.DataFrame(pca.components_)
pca_components.columns = idvs

pca_features = pd.DataFrame(pca.fit_transform(title[idvs]))
pca_features.columns = ['PC1','PC2']



