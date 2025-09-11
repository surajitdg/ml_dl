import numpy as np




class LinReg():

    def __init__(self, learning_rate, use_grad_desc, iters):
        self.learning_rate = learning_rate
        self.use_grad_desc = use_grad_desc
        self.iters = iters

    def fit(self, X, y):
        self.w = np.random.uniform(0,1,size=(X.shape[1]))
        self.b = 0
        if self.use_grad_desc:
            #grad descent implementation
            for i in range(self.iters):
                y_pred = np.dot(X,self.w)+self.b #predict y

                #print metrics
                rmse = np.sqrt(np.mean(np.pow(y_pred-y, 2)))
                print (f'Iteration {i} loss is {rmse}')

                #gradient descent
                dw = np.mean(np.dot(np.transpose(X),(y_pred-y)))
                db = np.mean(np.sum(y_pred-y))

                self.w = self.w - self.learning_rate * dw
                self.b = self.b - self.learning_rate * db



        else:
            # (X_T.X)^-1.X_T
            X = np.add(np.ones((X.shape[0],1)),X) #add bias first
            sym_x = np.dot(np.transpose(X),X) #sym matrix
            pseudo_inv = np.dot(np.linalg.inv(sym_x),np.transpose(X)) #moore-penrose pseudoinverse
            self.w = np.dot(pseudo_inv, y)

    def predict(self, x):
        if self.use_grad_desc:
            out = np.dot(x,self.w)+ self.b
        else:
            x = np.add(np.ones((x.shape[0],1)),x)
            out = np.dot(x,self.w)

        return out


if __name__ == "__main__":
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10]) 
    lr = LinReg(learning_rate=0.01, use_grad_desc=True, iters=30)
    lr.fit(X,y)
    print("GD Weights:", lr.w, "Bias:", lr.b)
    print("GD Predictions:", lr.predict(np.array([[6]])))

    lr2 = LinReg(learning_rate=None, use_grad_desc=False, iters=None)
    lr2.fit(X,y)
    print("PINV Weights:", lr2.w, "Bias:", lr2.b)
    print("PINV Predictions:", lr2.predict(np.array([[6]])))

        

            