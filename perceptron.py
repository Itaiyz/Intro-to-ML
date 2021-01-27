#################################
# Your name: Itai Zemah
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import matplotlib.pyplot as plt
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing

"""
Assignment 3 question 1 skeleton.

Please use the provided function signature for the perceptron implementation.
Feel free to add functions and other code, and submit this file with the name perceptron.py
"""

def helper():
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

def perceptron(data, labels):
        T= len(data)
        n=784
        weights= np.zeros(n)
        normal_data = sklearn.preprocessing.normalize(data)
        for t in range(T):
                y_pred= getPred(np.dot(weights,data[t]))
                if y_pred!= labels[t]:
                        weights= weights+ np.dot(labels[t],data[t])
        return np.array(weights)



                       

#################################

# Place for additional code

def getPred(vec):
        #print(len(vec))
        if vec<0:
                return -1
        return 1

def section_a(runs=100, lengths=[5,10,50,100,500,1000,5000]):
        train_data, train_labels, validation_data, validation_labels, test_data, test_labels= helper()
        values= []
        for leng in lengths:
                print("leng: "+str(leng)+" began")
                accuracies=[]
                for i in range(runs):
                        #print("leng: "+str(leng)+", i="+str(i))
                        joint= np.column_stack((train_data[:leng], train_labels[:leng]))
                        np.random.shuffle(joint)
                        
                        weights= perceptron(joint[:,0:784],joint[:,784])
                        accuracies.append(calc_accuracy(weights, test_data, test_labels))
                ins = [np.mean(accuracies), np.percentile(accuracies,5) , np.percentile(accuracies,95)]
                values.append(np.round(ins,4))      
                print("leng: "+str(leng)+" ended") 
        #plt.plot(accuracies.keys(), accuracies.items())
        #plt.xlabel("n")
        #plt.ylabel("Average accuracy")
        cols = ['accuracy average', '5% percentile', '95% percentile']
        colors1 = ['c','green', 'blue', 'pink']
        colors2 = ['c','green', 'blue', 'pink','c','green', 'blue']
        table = plt.table(cellText=values, rowLabels=lengths, rowColours=colors2,colColours=colors1, colLabels=cols, loc='center right')
        table.auto_set_font_size(False)
        table.set_fontsize(18)
        table.scale(3, 3)
        #plt.show()

def calc_accuracy(weights, test_data, test_labels):
        assert len(test_data)==len(test_labels)
        n= len(test_data)
        err=0
        for i in range(n):
                pred= getPred(np.dot(weights,test_data[i]))
                if pred!=test_labels[i]:
                        err+=1
        err/=n
        return 1-err


def section_b():
        #print("fetching data")
        train_data, train_labels, validation_data, validation_labels, test_data, test_labels= helper()
        weights= perceptron(train_data, train_labels)
        plt.imshow(np.reshape(weights, (28,28)), interpolation='nearest')
        #plt.show()

def section_c():
        train_data, train_labels, validation_data, validation_labels, test_data, test_labels= helper()
        weights= perceptron(train_data, train_labels)
        res= calc_accuracy(weights, test_data, test_labels)
        return res
        
def section_d():
        
        def calc_accuracy4(weights, test_data, test_labels):
                assert len(test_data)==len(test_labels)
                n= len(test_data)
                err=0
                wrong_labels= []
                for i in range(n):
                        pred= getPred(np.dot(weights,test_data[i]))
                        if pred!=test_labels[i]:
                                err+=1
                                if len(wrong_labels)<2:
                                        wrong_labels.append(test_data[i])
                err/=n
                return (1-err, wrong_labels)
        train_data, train_labels, validation_data, validation_labels, test_data, test_labels= helper()
        weights= perceptron(train_data, train_labels)
        accuracy, arr= calc_accuracy4(weights, test_data, test_labels)
        print(accuracy)
        plt.imshow( np.reshape(arr[1], (28,28)),interpolation='nearest')
        #plt.show()
        

if __name__ == '__main__':
        #section_a()
        #section_b()
        #section_d()

#################################
