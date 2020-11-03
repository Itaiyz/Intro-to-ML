def exec1():
    #A function that calls the essential data for the task
    from sklearn.datasets import fetch_openml
    mnist= fetch_openml('mnist_784')
    data= mnist['data']
    labels= mnist['target']
    import numpy.random
    idx= numpy.random.RandomState(0).choice(70000, 11000)
    train= data[idx[:10000], :].astype(int)
    train_labels = labels[idx[:10000]]

    test = data[idx[10000:], :].astype(int)
    test_labels = labels[idx[10000:]]

    return {'mnist':mnist, 'data':data, 'labels':labels, 'idx':idx, 'train':train, 'train_labels':train_labels,
            'test':test,'test_labels':test_labels}


                   

def kNNAlgo(train, labels, query, k):
    n=len(train)
    assert k<len(train)
    #calculating distance to each matrix
    from numpy import linalg as LA
    #calculating norms for each distance matrix
    distances= [LA.norm(train[i]-query) for i in range(n)]
    minDist= sorted(distances)[:k:]
    labels= [x for _,x in sorted(zip(distances,labels))][:k:]
    #finding the k smallest
    #distances= distances[:k]
    clusters= {val:0 for val in labels}
    for v in labels:
                    clusters[v]+=1
    import operator
    return max(clusters.items(),key=operator.itemgetter(1))[0]




    
    


    
