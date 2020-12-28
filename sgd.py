#################################
# Your name: Itai Zemah
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import numpy.random
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import sklearn.preprocessing

"""
Assignment 3 question 2 skeleton.

Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""

def helper_hinge():
        mnist = fetch_openml('mnist_784')
        data = mnist['data']
        labels = mnist['target']

        neg, pos = "0", "8"
        train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
        test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

        train_data_unscaled = data[train_idx[:6000], :].astype(float)
        train_labels = (labels[train_idx[:6000]] == pos)*2-1

        validation_data_unscaled = data[train_idx[6000:], :].astype(float)
        validation_labels = (labels[train_idx[6000:]] == pos)*2-1

        test_data_unscaled = data[60000+test_idx, :].astype(float)
        test_labels = (labels[60000+test_idx] == pos)*2-1

        # Preprocessing
        train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
        validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
        test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
        return train_data, train_labels, validation_data, validation_labels, test_data, test_labels

def helper_ce():
        mnist = fetch_openml('mnist_784')
        data = mnist['data']
        labels = mnist['target']
        
        train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:8000] != 'a'))[0])
        test_idx = numpy.random.RandomState(0).permutation(np.where((labels[8000:10000] != 'a'))[0])

        train_data_unscaled = data[train_idx[:6000], :].astype(float)
        train_labels = labels[train_idx[:6000]]

        validation_data_unscaled = data[train_idx[6000:8000], :].astype(float)
        validation_labels = labels[train_idx[6000:8000]]

        test_data_unscaled = data[8000+test_idx, :].astype(float)
        test_labels = labels[8000+test_idx]

        # Preprocessing
        train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
        validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
        test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
        return train_data, train_labels, validation_data, validation_labels, test_data, test_labels

def SGD_hinge(data, labels, C, eta_0, T):
        #Implements Hinge loss using SGD.
        # w = np.random.rand(data[0].shape[0])
        w= np.zeros(data[0].shape[0])
        n=len(data)
        for t in range(1, T+1):
                eta_t = eta_0 / t
                i = int(np.random.uniform(low=0, high=n))
                x= data[i]
                y=labels[i]
                if (np.dot(w, x)*y < 1):
                        w = (1-eta_t)*w + eta_t*C*y*x
                else:
                        w = (1-eta_t)*w

        return w


def SGD_ce(data, labels, eta_0, T):
        """
        Implements multi-class cross entropy loss using SGD.
        """
        w = np.zeros((10,len(data[0])))
        eta_t=eta_0
        for t in range(1, T+1):
                i = int(np.random.uniform(low=0, high=len(data)))
                grads = calc_grads(w, data[i], labels[i],10)
                eta_t/=t
                for j in range(10):
                        w[j] = w[j] - grads[j]*eta_0

        return w
#################################

# Place for additional code

#################################

def run_hinge(min_scale=-5, max_scale=4, C=1, T=1000, param= "eta", eta_0=1):
        X=[i for i in range(min_scale, max_scale+1)]
       
        Y=[]
        temp_mean=0
        train_data, train_labels, validation_data, validation_labels, test_data, test_labels= helper_hinge()
        np.seterr(all='ignore') #Circumventing the overflow error
        for i in range(min_scale, max_scale+1):
                temp_mean=0
                for j in range(10):
                        if param=="eta":
                                resVec=  SGD_hinge(train_data, train_labels, C, 10**i, T)
                                temp_mean+=get_accuracy(validation_data, validation_labels, resVec)
                        if param=="C":
                                resVec=  SGD_hinge(train_data, train_labels, 10**i, eta_0, T)
                                temp_mean+=get_accuracy(validation_data, validation_labels, resVec)
                temp_mean=temp_mean/10
                Y.append(temp_mean)
        
        assert len(X)==len(Y)
        plt.clf()
        plt.scatter(X, Y)
        plt.xlim(min_scale-1,max_scale+1)
        plt.ylim(-0.1,1.1)
        #fig,ax= plt.subplots()
        #for x, y in zip(X, Y):
        #        ax.annotate("{:.2f},{:.2f}".format(x, y), xy=(x, y))
        plt.xlabel("log_10 of C values")
        plt.ylabel("Average accuaracy")
        #plt.show()


def run_ce(min_scale=-5, max_scale=4, T=1000, eta_0=1):
        #X=[i for i in range(min_scale, max_scale+1)]
        X= [10**(-1)+ 10**(-2)*i for i in range(-5,4)] 
        Y=[]
        temp_mean=0
        train_data, train_labels, validation_data, validation_labels, test_data, test_labels= helper_ce()
        for x in X:
                temp_mean=0
                for j in range(10):
                        resVec=  SGD_ce(train_data, train_labels, 10**x, T)
                        temp_mean+=get_ce_accuracy(validation_data, validation_labels, resVec)
                        
                temp_mean=temp_mean/10
                Y.append(temp_mean)
        
        assert len(X)==len(Y)
        plt.clf()
        plt.scatter(X, Y)
        #plt.xlim(min_scale-1,max_scale+1)
        #plt.ylim(0.75,0.85)
        plt.xlabel("eta0 values: High resolution")
        plt.ylabel("Average accuaracy")
        #plt.show()


def get_accuracy(data, labels, weights):
        accuracy = 0
        for x, y in zip(data, labels):
                if (np.dot(weights, x)*y >= 1):
                        accuracy += 1

        accuracy /= min(len(data), len(labels))
        return accuracy

def get_ce_accuracy(data, labels, weights):
        assert len(data)==len(labels)
        n= len(data)
        res=0
        for x, y in zip(data, labels):
                prods = np.dot(weights,x)
                y_hat = np.argmax(prods)
                if y_hat == int(y):
                        res += 1
        return res/n

def draw_weights(best_eta=1, best_c=0.01, T=20000):
        train_data, train_labels, validation_data, validation_labels, test_data, test_labels= helper_hinge()
        weights = SGD_hinge(train_data, train_labels, best_eta, best_c, T)
        accuracy = get_accuracy(test_data, test_labels, weights)
        #plt.imshow(np.reshape(weights, (-1, 28)), interpolation='nearest')
        plt.title(f"accuracy:{accuracy}")
        #plt.show()




def calc_grads(classifiers, x, y,n):   
    """ 
    returns the gradient for ce loss
    
    """
    
    def calc_soft_max(classifiers, x, overflowFix=True):
        products = [np.dot(x, classifiers[i]) for i in range(10)]
        maxVal = np.max(products)
        products =  products - maxVal  #preveting np.exp from overflow
        exp_dots = np.exp(products)
        sumExp= np.sum(exp_dots)
        softMax = exp_dots / sumExp
        return softMax
            
    soft_max = calc_soft_max(classifiers, x)
    label = int(y)
    soft_max[label] = soft_max[label] - 1   
    gradients = []
    for i in range(n):
      gradients.append(soft_max[i] * x)
      
    return gradients

def plot_ce( eta=0.75, T=20000):
        train_data, train_labels, validation_data, validation_labels, test_data, test_labels= helper_ce()
        w = SGD_ce(train_data, train_labels, eta, T)
        accuracy = get_ce_accuracy(validation_data, validation_labels, w)
        for i, vec in enumerate(w):
                plt.title(f"w{i}, P:{accuracy}")
                #plt.imshow(np.reshape(vec, (-1, 28)), interpolation='nearest')
                #plt.savefig(f"w{i}.png")
                #plt.show()


if __name__ == '__main__':
        #run_hinge(param="eta")
        plt.clf()
        #run_hinge(param="C")
        #draw_weights()
        #run_ce()
        #plot_ce()
        
