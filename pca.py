import numpy as np
from sklearn.decomposition import PCA as pc


class PCA():
    def __init__(self, components=10):
        """
        components: no of principal components
        
        """
        self.components = components

    
    def fit(self, dataset):
        dataset_t = np.transpose(dataset)
        d_dot_dt = np.dot(dataset, dataset_t)
        dt_dot_d = np.dot(dataset_t, dataset)

        # row space
        eigen_values,eigen_vectors_u  = np.linalg.eigh(d_dot_dt)
        idx = np.argsort(-eigen_values)
        eigen_values = eigen_values[idx[:self.components]]


        #columns space
        eigen_values2 , eigen_vectors_v = np.linalg.eigh(dt_dot_d)
        idx2 = np.argsort(-eigen_values2)
        total_sum_variance = sum(eigen_values2)
        self.ratios = (eigen_values2/total_sum_variance)*100
        print ('ratio of variances', self.ratios)
        print ('indexes with highest variance --',idx2)
        eigen_vectors_u = eigen_vectors_u[:,idx[:self.components]]
        # print (eigen_values, eigen_vectors_u)
        eigen_vectors_v = eigen_vectors_v[:, idx2[:self.components]]
        sigma = np.zeros(shape=(self.components, self.components))
        for i in range(0,self.components):
            for j in range(0, self.components):
                if i == j:
                    sigma[i][j] = np.sqrt(eigen_values2[i]).real

        self.weight = eigen_vectors_v
        return eigen_vectors_u , sigma, eigen_vectors_v
    
    def transform(self, dataset):
        return np.dot(dataset,self.weight)




if __name__ == "__main__":
    pca = PCA(components=2)
    dataset = [[0.1,0.3,0.5,0.6],
         [0.1,0.2,0.5,0.6],
         [0.1,0.3,0.9,0.6],
         [0.1,0.3,0.8,0.6],
         [0.2,0.2,0.5,0.6],
         [0.1,0.3,0.5,0.8],
         [0.2,0.3,0.5,0.6],
         [0.1,0.3,0.7,0.6],
         [0.1,0.5,0.5,0.6],
         [0.1,0.6,0.5,0.6],
         [0.41,0.41,0.41,0.41],
         [0.23,0.01,0.54,0.91],
         [0.11,0.21,0.58,0.09],
         [0.09,0.45,0.5,0.9],
         [0.01,0.32,0.99,0.72],
         [0.12,0.99,0.34,0.81],
         [0.89,0.25,0.53,0.05],
         [0.65,0.01,0.92,0.54],
         [0.03,0.81,0.5,0.9],
         ]
    # dataset = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    eigen_vectors_u , sigma, eigen_vectors_v = pca.fit(dataset)
    print (' U , sigma m , Vt', eigen_vectors_u , sigma, eigen_vectors_v)
    print ('transformed matrix in lower dimension space', pca.transform(dataset))
    # df = np.linalg.multi_dot([eigen_vectors_u, sigma, eigen_vectors_v])
    # print (df)
    print ("using pca from sklearn-----")
    pc2 = pc(n_components=2,svd_solver='full')
    pc2.fit(dataset)
    print (pc2.explained_variance_ratio_)
    print(pc2.fit_transform(dataset))
