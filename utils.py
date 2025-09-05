import numpy as np
import math


def entropy_categorical(y):
    """
    entropy = (-p log p), where p = count(feature)/total no of elements in the list
    """
    labels = list(set(y))
    entropy = 0
    for l in labels:
        p = y.count(l)/len(y)
        entropy = entropy+(-p * math.log(p, 2))
    return entropy

def entropy_weighted(df):
    """
    expected df['ind_features,'dependent']
    """
    ind_col = df.columns[0]
    x = list(df[ind_col])
    labels = list(set(x))
    weighted_entropy = 0
    for l in labels:
        y_df = df[df[ind_col] == l]
        y_total = list(y_df['y'])
        yc = list(set(y_total))
        entropy_weights = 0
        for d in yc:
            p = y_total.count(d)/len(y_total)
            entropy_weights = entropy_weights+(-p*math.log(p,2))
        l_prob = x.count(l)/len(x)
        weighted_entropy = weighted_entropy + (l_prob*entropy_weights)
    return weighted_entropy





