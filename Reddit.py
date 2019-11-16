# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from sklearn.manifold import TSNE
from sklearn.decomposition import LatentDirichletAllocation

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
title['YEAR'] = title.TIMESTAMP.astype(str).str[:4]
title['MONTH'] = title.TIMESTAMP.astype(str).str[5:7]
title['DAY'] = title.TIMESTAMP.astype(str).str[8:10]
title['DAYOFWEEK'] = title.TIMESTAMP.dt.dayofweek.astype(str)
title['HOUR'] = title.TIMESTAMP.astype(str).str[11:13]

### Dataset Summaries
summary = title.describe().T
summary_noconflicts = title[title.LINK_SENTIMENT == 1].describe().T
summary_conflicts = title[title.LINK_SENTIMENT != 1].describe().T

### Variable Standardization
title = title.reset_index()

features = [x for x in numerical_cols if x != 'LINK_SENTIMENT']
cat_features = ['MONTH','HOUR','DAYOFWEEK','DAY']
cat_features_df = pd.get_dummies(title[cat_features])

title = title.join(cat_features_df)

#features = features + cat_features_df.columns.tolist()

for f in features:
    #Standardization
    title[f] = (title[f] - title[f].min())/(title[f].max() - title[f].min())
    #Normalization
    #title[f] = (title[f] - title[f].mean())/np.std(title[f])

# =============================================================================
# ### Initial clusters
# idvs = ['LINK_SENTIMENT','Num_words','Frac_alphabet','Frac_digits','Frac_special','Avg_wordlength',
#         'Readability_index','Compound_sentiment','LIWC_Funct','LIWC_Negate','LIWC_Swear','LIWC_Social',
#         'LIWC_Affect','LIWC_CogMech','LIWC_Percept','LIWC_Bio','LIWC_Relativ','LIWC_Money','DAYOFWEEK','HOUR']
# 
# kmeans = KMeans(n_clusters=10, random_state=0, n_jobs=-1).fit(title[idvs])
# 
# title['Initial_Clusters'] = kmeans.labels_
# 
# cluster_characteristics = pd.DataFrame(kmeans.cluster_centers_)
# cluster_characteristics = cluster_characteristics.join(title.Initial_Clusters.value_counts())
# cluster_characteristics.columns = idvs + ['Row_Count']
# =============================================================================

### PCA
pca = PCA(n_components=len(features), random_state=0).fit(title[features])
pca_components = pd.DataFrame(pca.components_.T*np.sqrt(pca.explained_variance_)).T
pca_components.columns = features
pca_components = pca_components.T
eigenValues = pca.explained_variance_ratio_

#pca_features = pd.DataFrame(pca.fit_transform(title[features]))

### Factor Analysis
fa = FactorAnalysis(n_components=len(features), random_state=0).fit(title[features])
fa_components = pd.DataFrame(fa.components_)
fa_components.columns = features
fa_components = fa_components.T

### TSNE
tsne = TSNE(n_components=3, random_state=0).fit(title[features])

### LDA

lda = LatentDirichletAllocation(n_components=10, random_state=0, n_jobs=-1).fit(title[features[21:]])
lda_fit_transform = lda.fit_transform(title[features[21:]])

