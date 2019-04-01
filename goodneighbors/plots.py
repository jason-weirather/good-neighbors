from plotnine import *
from scipy.cluster.hierarchy import leaves_list
from scipy.cluster.hierarchy import linkage
import pandas as pd
def plot_counts_per_radius(census):
    return (ggplot(census,aes(x='distance'))
     + geom_boxplot(aes(ymin='0.05',lower='0.25',middle='0.5',upper='0.75',ymax='0.95'),
                stat='identity')
     + theme_bw()
    )

def plot_kmeans_evaluation(values):
    return (ggplot(values,aes(x='k',y='inertia'))
     + geom_line()
     + theme_bw()
    )

def plot_cluster_composition(counts,label,autosort):
    mat = counts.pivot(columns='phenotype_label',index=label,values='fraction')
    lorder = leaves_list(linkage(mat,method='ward',optimal_ordering=True))
    lorder = mat.iloc[lorder].index
    temp = counts.copy()
    if autosort: temp[label] = pd.Categorical(temp[label],categories=lorder)
    g = (ggplot(temp,aes(x=label,y='fraction',fill='phenotype_label'))
     + geom_bar(stat='identity')
     + theme_bw()
     + theme(axis_text_x=element_text(rotation=90, hjust=0.5))
    )
    return g
