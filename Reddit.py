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
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd
from sklearn.manifold import TSNE
import umap
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
import joblib

def cluster_scores(df,feature_name):
    print(silhouette_score(df[f_features], df[feature_name], random_state=0, metric='euclidean'))
    print(davies_bouldin_score(df[f_features],df[feature_name]))

title = DataFrame.from_csv("Reddit_Title.tsv", sep="\t")

title = title.reset_index()

### DATA PREPARATION
sr_once = title.SOURCE_SUBREDDIT.value_counts()[title.SOURCE_SUBREDDIT.value_counts() == 1].index.tolist()
title = title[title.SOURCE_SUBREDDIT.isin(sr_once)]

title = title.sample(10000, random_state = 3)

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

### Final Features - RUN FROM HERE
title['LIWC_Neg_New'] = title[['LIWC_Negemo',
'LIWC_Anx',
'LIWC_Anger',
'LIWC_Sad']].mean(axis = 1)
f_features = ['LIWC_Swear','LIWC_Social','LIWC_Neg_New','LIWC_Posemo','LIWC_Percept',
              'LIWC_Bio','LIWC_Relativ','LIWC_Work','LIWC_Achiev','LIWC_Leisure',
              'LIWC_Home','LIWC_Money','LIWC_Relig','LIWC_Death']


### Dataset Summaries
summary = title.describe().T
summary_noconflicts = title[title.LINK_SENTIMENT == 1].describe().T
summary_conflicts = title[title.LINK_SENTIMENT != 1].describe().T

### Points that have been labelled

labelled_pts = ['djiosmo',
'fakeid2',
'openelec',
'teamearth',
'moviefunfacts',
'ajestuncon',
'shin',
'plazaaragon',
'dogestarter',
'letsdub',
'nocrychallenge',
'vocalists',
'complexsystems',
'the_scoundrealm',
'pensacolabeer',
'misconceptionfixer',
'originalerror',
'wrestlingpod',
'nightshade',
'ihscout']

### Variable Standardization
title = title.reset_index()

features = [x for x in numerical_cols if x != 'LINK_SENTIMENT']
cat_features = ['MONTH','HOUR','DAYOFWEEK','DAY']
cat_features_df = pd.get_dummies(title[cat_features])

title = title.join(cat_features_df)

for f in f_features:
    #Standardization
    title[f] = (title[f] - title[f].min())/(title[f].max() - title[f].min())
    #Normalization
    #title[f] = (title[f] - title[f].mean())/np.std(title[f])

### ONLY FOR SAMPLING
#title = title.sample(10000)
title_all = pd.DataFrame()
for th in [.99]: # .95,.9,.85 - other thresholds to test
    title_sw = title[(title.LIWC_Swear >= title.LIWC_Swear.quantile(th))&
                  (title.LIWC_Neg_New >= title.LIWC_Neg_New.quantile(th))|
                  (title.LIWC_Swear >= title.LIWC_Swear.quantile(th))&
                  (title.LIWC_Bio >= title.LIWC_Bio.quantile(th))]
    print(th)
    print('title_sw shape:' + str(title_sw.shape))
    for col in ['LIWC_Social','LIWC_Posemo','LIWC_Percept','LIWC_Relativ',
'LIWC_Work','LIWC_Achiev','LIWC_Leisure','LIWC_Money','LIWC_Relig','LIWC_Death','LIWC_Home']:
        if (col in ['LIWC_Relig','LIWC_Death','LIWC_Home'])&(th < 0.97):
            th = 0.97
        else:
            pass
        title_oth = title[title[col] >= title[col].quantile(th)]
        print(col + ' shape: ' + str(title_oth.shape))
        title_all = title_all.append(title_oth)
    title_all = title_all.append(title_sw)
    title_all = title_all.drop_duplicates()
    print('final title shape: ' + str(title_all.shape))
    print('\n')

title = title_all.copy()
del [title_oth, title_sw, title_all, cat_features_df]

title = title.reset_index()

#features = features + cat_features_df.columns.tolist()



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

### Getting number of features to use - Initially used all 14 components, took only those explaining > 90% data
pca2 = PCA(n_components= len(f_features), random_state=0).fit(title[f_features])
pca2_components = pd.DataFrame(pca2.components_.T*np.sqrt(pca2.explained_variance_)).T
pca2_components.columns = f_features
pca2_components = pca2_components.T
eigenValues = pca2.explained_variance_ratio_

pca_transformed_features = pd.DataFrame(pca2.fit_transform(title[f_features]))
pca_transformed_features.columns = ['PC_' + str(i + 1) for i in range(len(f_features))]
pca2_components.columns = ['PC_' + str(i + 1) for i in range(pca2_components.shape[1])]
### CHANGE THIS IF DATA IS SAMPLED
pc_to_remove = ['PC_13','PC_14']
pca_transformed_features = pca_transformed_features.drop(pc_to_remove, axis = 1)

title = title.join(pca_transformed_features)

f_features = pca_transformed_features.columns.tolist()
del pca_transformed_features

### LDA - finding optimal number of components
'''
perp2 = []
for i in [1,2,3,4,5,6,7,8,9,10,15,20,25,30,50,100]:
    lda = LatentDirichletAllocation(n_components=i, random_state=0, n_jobs=-1).fit(title[f_features])
    x = lda.perplexity(title[f_features])
    print(x)
    perp2.append(x)
    print(i)
'''
    
#lda = LatentDirichletAllocation(n_components=10, random_state=0, n_jobs=-1).fit(title[f_features])
#lda_fit_transform = lda.fit_transform(title[f_features])

### k-means - # Optimal Clusters = 7 (SSE Threshold = 11,000)

title_sample = title.sample(10000, random_state = 3)
#.append(title[title['SOURCE_SUBREDDIT'].isin(labelled_pts)]).drop_duplicates()

sse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000, random_state=0, n_jobs=-1, init = 'k-means++').fit(title[f_features])
    title["kmeans_clusters_"+str(k)] = kmeans.labels_
    sse[k] = kmeans.inertia_
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()

num_clusters = 6
### Initial Clustering with full dataset = 7, after reducing rows based on quantiles = 6
kmeans_final = KMeans(n_clusters=num_clusters, max_iter=1000, random_state=0, n_jobs=-1, init = 'k-means++').fit(title_sample[f_features])

#check = title_sample[['SOURCE_SUBREDDIT','kmeans_clusters_'+str(num_clusters)]][title_sample['SOURCE_SUBREDDIT'].isin(labelled_pts)].drop_duplicates()

dict_cluster_results = {}
for c in title['kmeans_clusters_'+str(num_clusters)].unique().tolist():
    dict_cluster_results[c] = title[title['kmeans_clusters_'+str(num_clusters)] == c][f_features].describe().T

### Evaluating k-means
cluster_scores(title_sample,'kmeans_clusters_'+str(num_clusters))

### SPECTRAL CLUSTERING - ONLY ON SAMPLED DATA
sc = SpectralClustering(n_jobs=-1, random_state=0).fit(title[f_features])
title['sc_clusters'] = sc.labels_
cluster_scores(title,'sc_clusters')

### GMM
gmm = GaussianMixture(n_components = num_clusters, covariance_type = 'spherical',random_state= 0).fit(title[f_features])
gmm_pred_proba = pd.DataFrame(gmm.predict_proba(title[f_features]))
title['gmm_clusters'] = gmm.predict(title[f_features])

### 2nd best clusters
#title.loc[title['gmm_clusters'] == 2,'gmm_clusters_2'] = 5
#title.loc[title['gmm_clusters'] == 5,'gmm_clusters_2'] = 2
#title.loc[title['gmm_clusters_2'].isnull(),'gmm_clusters_2'] = title['gmm_clusters']

cluster_scores(title,'gmm_clusters')

### Hierarchical Clustering

for i in [2,3,4,5,6,7,8,9,10]:
    hac = AgglomerativeClustering(n_clusters=i, affinity='euclidean', linkage='average').fit(title[f_features])
    title['hac_clusters'] = hac.labels_
    print(i)
    cluster_scores(title,'hac_clusters')
    print('\n')

# DENDROGRAM
linked = linkage(title[f_features],'average')
labelList = range(title.shape[0])

plt.figure(figsize=(100,70))
dendrogram(linked, orientation = 'top', labels = labelList, distance_sort='descending',
           show_leaf_counts=True)
plt.ylim(0.6,1)
plt.xlim(0,np.round(title.shape[0]/10))
plt.show()

hac = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='average').fit(title[f_features])
title['hac_clusters'] = hac.labels_

dict_hac_results = {}
for c in title['hac_clusters'].unique().tolist():
    dict_hac_results[c] = title[title['hac_clusters'] == c][f_features].describe().T

joblib.dump(hac, 'Hierarchical_Model.pkl')

cluster_to_subreddit = {}
for c in title['hac_clusters'].unique():
    cluster_to_subreddit[c] = list(title[title['hac_clusters'] == c]['SOURCE_SUBREDDIT'].unique())

for c in title['hac_clusters'].unique():
    title.loc[title['TARGET_SUBREDDIT'].isin(cluster_to_subreddit[c]), 'hac_cluster_target'] = c
    
title = title[~title['hac_cluster_target'].isnull()]

### Anomaly Detection using SVD
f_features = ['LIWC_Swear','LIWC_Social','LIWC_Neg_New','LIWC_Posemo','LIWC_Percept',
              'LIWC_Bio','LIWC_Relativ','LIWC_Work','LIWC_Achiev','LIWC_Leisure',
              'LIWC_Home','LIWC_Money','LIWC_Relig','LIWC_Death']
u, s, v = np.linalg.svd(title[f_features])

title_2d = pd.DataFrame(u[:,:2])

### SPECTRAL CLUSTERING - ON THE FIRST TWO DIMENSIONS OF U
sc = SpectralClustering(n_jobs=-1, random_state=0, n_clusters=4).fit(title_2d)
title_2d['sc_clusters'] = sc.labels_
cluster_scores(title_2d,'sc_clusters')
    
fig, ax = plt.subplots()
ax.margins(0.05)
groups = title_2d.groupby('sc_clusters')
for name,group in groups:
    ax.plot(group[0], group[1])


### UMAP
reducer = umap.UMAP()
embedding = reducer.fit_transform(title_2d[[0,1]])
print(embedding.shape)

### Kmeans on UMAP data
embedding = pd.DataFrame(embedding)
f_features = embedding.columns.tolist()
sse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000, random_state=0, n_jobs=-1, init = 'k-means++').fit(embedding[f_features])
    embedding["kmeans_clusters_"+str(k)] = kmeans.labels_
    sse[k] = kmeans.inertia_
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()

num_clusters = 5
kmeans_final = KMeans(n_clusters=num_clusters, max_iter=1000, random_state=0, n_jobs=-1, init = 'k-means++').fit(embedding[f_features])
cluster_scores(embedding,'kmeans_clusters_'+str(num_clusters))

### INTERPRETING CLUSTERS
f_features = ['LIWC_Swear','LIWC_Social','LIWC_Neg_New','LIWC_Posemo','LIWC_Percept',
              'LIWC_Bio','LIWC_Relativ','LIWC_Work','LIWC_Achiev','LIWC_Leisure',
              'LIWC_Home','LIWC_Money','LIWC_Relig','LIWC_Death']

embedding_req = embedding.join(title[['SOURCE_SUBREDDIT','TARGET_SUBREDDIT','LINK_SENTIMENT'] + f_features].reset_index())
embedding_req = embedding_req.reset_index()
source_clusters = embedding_req[['SOURCE_SUBREDDIT','kmeans_clusters_5']].groupby('SOURCE_SUBREDDIT')['kmeans_clusters_5'].first().reset_index()
source_clusters.columns = ['TARGET_SUBREDDIT','kmeans_clusters_5_target']
embedding_req = embedding_req.merge(source_clusters, on = 'TARGET_SUBREDDIT', how = 'left')

embedding_req = embedding_req[~embedding_req.TARGET_SUBREDDIT.isin(embedding_req[embedding_req['kmeans_clusters_5_target'].isnull()]['TARGET_SUBREDDIT'].unique().tolist())]

conflicts = 1 - embedding_req.groupby(['kmeans_clusters_5','kmeans_clusters_5_target'])['LINK_SENTIMENT'].mean().reset_index()

### FINALIZING RESULTS

plt.scatter(embedding[0], embedding[1], c = embedding['kmeans_clusters_5'])
plt.legend(loc = 'best')

writer = pd.ExcelWriter('All_Cluster_Summaries.xlsx', engine='xlsxwriter')

summaries = {}
for i in embedding_req.kmeans_clusters_5.unique():
    summaries[i] = embedding_req[embedding_req['kmeans_clusters_5'] == i].describe().T
    summaries[i].to_excel(writer, sheet_name='Sheet' + str(i))
    
writer.save()

total_summary = embedding_req.describe().T
total_summary.to_excel('Feature_Distributions.xlsx')



