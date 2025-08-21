import numpy as np
import pandas as pd
from utils import entropy_categorical

class Node():
    def __init__(self, feature=0, threshold=0, value=0, left_subtree=None, right_subtree=None):
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.left_subtree = left_subtree
        self.right_subtree = right_subtree

class DecisionTree():

    def __init__(self, min_samples_split=2, min_impurity=0, max_depth=None, loss_function=None):
        """
        min_samples_split: min no of samples required for split
        min_impurity: min impurity for split
        max_depth: max depth to prune the tree
        loss_function: loss function to use entropy or gini coefficient
        
        """
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.loss_function = loss_function

    def build_tree(self, Xy):
        """
        Assume, X is pandas
        """
        X = Xy.drop('y',axis=1)
        no_of_samples, no_of_features = X.shape
        features = X.columns
        entropy_temp = {}
        y_entropy = entropy_categorical(list(Xy['y']))
        # print(y_entropy)
        gain = -999999
        select_features = 0
        if no_of_samples > self.min_samples_split and len(features) > 0:
            for i in range(0,no_of_features):
                entropy_temp[i] = entropy_categorical(list(X[features[i]]))
                # print (entropy_temp)
                temp_gain = y_entropy - entropy_temp[i]
                if temp_gain > gain:
                    gain = temp_gain
                    select_feature = i
                    best_feature = features[select_feature]
            print (f'total gain {gain} for best feature {best_feature}')
            f = list(set(list(X[best_feature])))
            #print (f)
            Xy_left = Xy[Xy[best_feature] == f[0]]
            Xy_right = Xy[Xy[best_feature] != f[0]]
            Xy_left_node = self.build_tree(Xy_left.drop(best_feature,axis=1))
            Xy_right_node = self.build_tree(Xy_right.drop(best_feature,axis=1))
            print (Xy_left_node.feature,Xy_right_node.feature)
            return Node(feature=best_feature,threshold=gain,left_subtree=Xy_left_node,right_subtree=Xy_right_node)
        else:
            y_s = list(Xy['y'])
            y_value = max(y_s.count(0),y_s.count(1))
            return Node(value=y_value)
    
    def fit(self, Xy):
        self.root = self.build_tree(Xy)
        return self.root

        
    def traverse(self,root):
        if root.left_subtree  == None and root.right_subtree == None:
            #print (root.feature)
            return None
        else:
            print (root.feature, root.threshold)
            self.traverse(root.left_subtree)
            self.traverse(root.right_subtree)



        



                       
        # print (entropy_temp)


if __name__ == "__main__":
    dt = DecisionTree()
    df = pd.DataFrame(data=[[1, 1, 1, 1],
                            [0, 0, 1, 0],
                            [1, 0, 1, 0], 
                            [1, 0, 1, 1], 
                            [1, 0, 1, 1],
                            [1, 0, 1, 1],
                            [1, 0, 0, 1], 
                            [0, 0, 0, 1], 
                            [1, 0, 0, 1], 
                            [0, 1, 0, 0]
                            ],columns=['first','second','third','y'])
    # y = [1, 0, 0, 1, 1, 1, 1, 1, 1, 0]
    base_node = dt.fit(df)
    dt.traverse(base_node)
    # print (dt.root.left_subtree)


                




    
