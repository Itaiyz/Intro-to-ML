#################################
# Your name: Itai Zemah
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib.
from matplotlib import pyplot as plt
import numpy as np
from process_data import parse_data

np.random.seed(7)


def run_adaboost(X_train, Y_train, T=80):
    """
    Returns: 

        hypotheses : 
            A list of T tuples describing the hypotheses chosen by the algorithm. 
            Each tuple has 3 elements (h_pred, h_index, h_theta), where h_pred is 
            the returned value (+1 or -1) if the count at index h_index is <= h_theta.

        alpha_vals : 
            A list of T float values, which are the alpha values obtained in every 
            iteration of the algorithm.
    """
    #initialization
    from math import log
    assert len(X_train)==len(Y_train)
    n= len(X_train)
    D= [1/n for j in range(n)]
    weights= [0]
    hypothesis= [()]
    #import sys
    #sys.setrecursionlimit(n)
    for t in range(1,T+1):
        #print(str(t)+"-started")
        h=get_WL(D, X_train, Y_train)
        hypothesis.append(h)
        #print("t-"+ str(t)+" length:"+str(len(hypothesis)))
        epsilon_t=calc_weighted_error(D, t, X_train,Y_train, h)
        weights.append( 0.5* np.log((1-epsilon_t)/epsilon_t))
        ekans = np.array([(-1)*weights[t]*Y_train[i]*get_h(h, x=X_train[i]) for i in range(n)])  # Ekans- the snake pokemon
        arbok= np.exp(ekans) #the evolution of ekans
        denom = np.multiply(D, arbok)
        D = denom / np.sum(denom)
        #print(str(t)+"-finished")
    #print(len(hypothesis))
    return hypothesis, weights    



##############################################
# You can add more methods here, if needed.

def get_WL(D, x_train, Y_train):
    """
    return weak learner function for adaboost
    """
    pos_j, pos_theta, Gpos= WL_wrapper(D, x_train, Y_train,1)
    neg_j, neg_theta, Gneg= WL_wrapper(D, x_train, Y_train, -1)
    if Gpos<Gneg:\
        return (1, pos_j, pos_theta)
    return (-1, neg_j, neg_theta)

def WL_wrapper(D, X_train, Y_train, tag):
    min_sumD= np.Infinity      #min error
    tetha = 0
    JMin = 0
    d = len(X_train[0])
    n = len(X_train)
    #print(n)
    #print(d)
        
    for j in range(d):
        arr=[]
        for i in range(n):
            #assert type(D[i])==np.float64 or type(D[i])==float
            arr.append([X_train[i][j], Y_train[i], D[i]])
        
        jVals=np.array(arr)
        jVals = jVals[np.argsort(jVals[:, 0])]
        mat_at_n1 = np.array([jVals[n-1][0]+1, 0,0])
        jVals= np.vstack((jVals, mat_at_n1))     # add x_m+1,j = x_m,y  +1
        sumD = 0
        for i in range(n):  #calc sum Di from rec where yi= sign(b)
            #remider: jVals[i][0]- xi, jVals[i][1]- yi, jVal[i][2]- D[i]
            x_i= jVals[i][0]
            y_i= jVals[i][1]
            D_i= jVals[i][2]
            if(y_i==tag):
                sumD += D_i
        try:
            if(sumD<min_sumD):
                min_sumD = sumD
                tetha = jVals[0][0]-1
                JMin = j
        except:
            print(type(D_i))
            print(type(sumD))
            print(len(sumD))
            print(type(min_sumD))
        for i in range(n): #choose tetha for h
            x_i= jVals[i][0]
            y_i= jVals[i][1]
            D_i= jVals[i][2]
            sumD = sumD - tag*y_i*D_i
            if(sumD<min_sumD and  i+1<n and x_i!=jVals[i+1][0]):
                min_sumD =  sumD
                tetha = 0.5 * (x_i + jVals[i+1][0])
                JMin = j
                
    return (JMin,tetha,  min_sumD )


    

    
    

def get_h(h, x):
    #print(type(x))
    #print(len(h))
    try:
        pred = h[0]
        ind = h[1]
        theta = h[2]
    except:
        print(len(h))
    if(x[int(ind)] <= theta):
        return pred
    return -1*pred

def calc_weighted_error(D, ind, X_train, Y_train, h):
    sum = 0
    n=len(X_train)
    for i in range(n):
        sum += D[i]*calc_z0_loss(X_train[i], Y_train[i], h)
    
    return sum



def calc_z0_loss(x, y, h):
    pred = h[0]
    index = h[1]
    theta = h[2]
    if(x[index] > theta):
        pred*=-1
    if pred==y:
        return 0
    return 1



def sign(num):
    if num!=0:
        return np.sign(num)
    return 1
def calc_error(data, labels, hypotheses, weights):
    """
    calc empirical error
    classfication is sign of sum of ai*hi(x)
    """
    errors= []
    n = len(data)
    sumT = np.zeros(n)    
    T = len(hypotheses)
    for t in range(1,T):
        err = 0
        for i in range(n):
            sumT[i] += weights[t] * get_h(hypotheses[t], data[i])
            if(sign(sumT[i])!=labels[i]):
                err += 1
        errors.append(err/n)


    return errors



    
    

def calc_loss(data, labels, hypotheses, weights):
    """
    calc loss by T
    1/m * sum of exp(-yi sum(at*ht(x)))
    """

    losses= []
    n = len(data)
    expT = np.zeros(n)    
    T = len(hypotheses)
    for t in range(1,T):
        for i in range(n):
            expT[i] += (-1)*labels[i]*weights[t] * get_h(hypotheses[t], data[i])
        bonsly = np.exp(expT)
        sudowoodo = sum(bonsly)
        losses.append((1/n)*sudowoodo)


    return losses





##############################################


def main():
    data = parse_data()
    if not data:
        return
    (X_train, y_train, X_test, y_test, vocab) = data
    T=80
    #print("reminder- T should be changed")

    hypothesis, alpha_vals = run_adaboost(X_train, y_train, T)

    ##############################################
    # You can add more methods here, if needed.
  
    def plot_test_err(X_train= data[0], Y_train=data[1], X_test=data[2], Y_test=data[3], T=80):
        err_train= calc_error(X_train, Y_train, hypothesis, alpha_vals)
        err_test= calc_error(X_test, Y_test, hypothesis, alpha_vals)
        plt.xlabel("t values")
        plt.ylabel("error")
        plt.plot([t for t in range(1,T+1)],err_train, color= 'green' ,label='train error')
        plt.plot([t for t in range(1,T+1)],err_test, color='red', label='test error')
        plt.legend(loc="upper right")
        #plt.show()

    def get_predictors(X_train=data[0], Y_train= data[1], vocab= data[4], T=80):
        for t in range(1,len(hypothesis)):
            #print(hypothesis[t])
            #print("index "+str(hypothesis[t][1])+": "+vocab[hypothesis[t][1]])
    def plot_loss_func(X_train= data[0], Y_train=data[1], X_test=data[2], Y_test=data[3], T=80):
        loss_train= calc_loss(X_train, Y_train, hypothesis, alpha_vals)
        loss_test= calc_loss(X_test, Y_test, hypothesis, alpha_vals)
        plt.xlabel("t values")
        plt.ylabel("loss")
        X=[t for t in range(1,T+1)]
        plt.plot(X,loss_train, color= 'green' ,label='train loss')
        plt.plot(X,loss_test, color='red', label='test loss')
        plt.legend(loc="upper right")
        #plt.show()
    #plot_test_err()
    #get_predictors()
    #plot_loss_func()
    



    ##############################################

if __name__ == '__main__':
    main()



