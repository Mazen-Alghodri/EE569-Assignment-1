import numpy as np
import matplotlib.pyplot as plt
class Linear: #Defining the layer that computes Y=Ax+B
    def __init__(self,a,b, learningrate=0.1):
        self.loss=None
        self.a=a
        self.x=None
        self.predy=None
        self.b=b
        self.learningrate=learningrate
    def forward(self,x):
        self.x=x #Storing X , need that for gradient since DA/Dy=x
        self.predy= np.dot(self.a,self.x)+self.b #Computing Y=Ax+b where Y is our predicted Value(s)
    def backward(self,grady):
        x = np.atleast_2d(self.x)
        if grady.ndim == 1 or grady.shape[1] == 1: #grad a is vector
            grada = np.outer(grady.ravel(), x.ravel()) #Da/Dy=x , for backpropagation we multiply this by the previous gradient(s)
            gradb = grady.ravel() #Db/dy = 1 , we basically just pass the gradient down
        else: #Grady is matrix
            m = grady.shape[1]
            grada = np.dot(grady, x.T) / m #averaging gradient across batch for mean loss
            gradb = np.mean(grady, axis=1) #averaging bias gradient
        self.a -= self.learningrate * grada #updating A
        self.b -= self.learningrate * gradb#updating b
def MSE(ypred,y):
    return 2 * (ypred - y) #MSE gradient is only used for single test of the linear function.
class sigmoid: #Sigmoid is implemented to use this class for logistic regression
    def __init__(self):
        self.sigforward=None
    def forward(self,ylinear):
        self.sigforward= 1/(1+np.exp(-ylinear))
    def backward(self,backprop):
        return backprop*self.sigforward * (1-self.sigforward)
class bce: #BCE is implemented to use this class for logistic regression
    def __init__(self):
        self.ypred=None
        self.ytrue=None
        self.loss=None
    def forward(self,ypred,ytrue):
        self.ypred=ypred
        self.ytrue=np.transpose(ytrue)
        self.loss=-np.mean(ytrue*np.log(ypred) + (1-ytrue)*np.log(1-ypred))
    def backward(self):
        return (self.ypred - self.ytrue) / (self.ypred * (1 - self.ypred))
if __name__=="__main__":
    #Setting test weights and input/output to train
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([0.5, 0.5])
    x = np.array([1.0, 2.0])
    ytrue = np.array([2.0, 3.0])
    linear = Linear(A, b , 0.1)#creating the computation node
    epoch=10#training for only 10epochs
    losses=[]
    for i in range(epoch):#Training
        linear.forward(x)
        ypred=linear.predy
        grady=MSE(ypred,ytrue)
        linear.backward(grady)
        losses.append(np.mean((ypred-ytrue)**2))
    #Plot Part for Loss curve and test on a single prediction
    plt.plot(losses)
    plt.title('Loss curve')
    plt.show()
    plt.scatter(ytrue[0], ytrue[1], c='blue', label='True')
    plt.scatter(linear.predy[0], linear.predy[1], c='red', label='Predicted')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.legend()
    plt.show()
    print(linear.predy,ytrue)
