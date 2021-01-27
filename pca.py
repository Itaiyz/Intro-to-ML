import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
import numpy as np
from numpy import matrix as mat
from numpy import linalg as lin
from sklearn import svm
from sklearn.model_selection import train_test_split

def plot_vector_as_image(image, h, w):
        """
        utility function to plot a vector as image.
        Args:
        image - vector of pixels
        h, w - dimesnions of original pi
        """     
        plt.imshow(image.reshape((h, w)), cmap=plt.cm.gray)
        plt.title(title, size=12)
        plt.show()

def get_pictures_by_name(name='Ariel Sharon'):
        """
        Given a name returns all the pictures of the person with this specific name.
        YOU CAN CHANGE THIS FUNCTION!
        THIS IS JUST AN EXAMPLE, FEEL FREE TO CHANGE IT!
        """
        lfw_people = load_data()
        selected_images = []
        n_samples, h, w = lfw_people.images.shape
        target_label = list(lfw_people.target_names).index(name)
        for image, target in zip(lfw_people.images, lfw_people.target):
                if (target == target_label):
                        image_vector = image.reshape((h*w, 1))
                        selected_images.append(image_vector)
        return selected_images, h, w

def load_data():
        # Don't change the resize factor!!!
        lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
        return lfw_people

######################################################################################
"""
Other then the PCA function below the rest of the functions are yours to change.
"""

def PCA(X, k):
        """
        Compute PCA on the given matrix.

        Args:
                X - Matrix of dimesions (n,d). Where n is the number of sample points and d is the dimension of each sample.
                For example, if we have 10 pictures and each picture is a vector of 100 pixels then the dimesion of the matrix would be (10,100).
                k - number of eigenvectors to return

        Returns:
          U - Matrix with dimension (k,d). The matrix should be composed out of k eigenvectors corresponding to the largest k eigenvectors 
                        of the covariance matrix.
          S - k largest eigenvalues of the covariance matrix. vector of dimension (k, 1)
        """
        
        cov_id_vaccine= np.matmul(mat.transpose(X),X)
        w, v= lin.eig(cov_id_vaccine)
        v = mat.transpose(v)
        vec= []
        d= len(w)
        for i in range(d):
                vec.append((w[i],v[i]))
        vec.sort(key= lambda x: x[0], reverse=True)
        
        U = np.stack([vec[i][1] for i in range(k)], axis=0)
        #print(type(w))
        w[::-1].sort()
        w=w[:k]
        S = np.array([w]).T
        return U, S


def standardize(data):
        d= len(data[0])
        n= len(data)
        res = np.zeros((1, d))
        for i in range(n):
                res = res + data[i]
        res = res / n
        for i in range(n):
                data[i] = data[i] - res
        

def section_b():
        selected_images, h, w = get_pictures_by_name()
        #print("got here")
        data = np.array(selected_images)[:, :, 0]
        #standardize(data)
        U, S = PCA(data, 10)
        plot_vectors(U, h, w, 2, 5)

def section_c(k_vals=[1,5,10,30,50,100,150,300]):
    selected_images, h, w = get_pictures_by_name()
    data = np.array(selected_images)[:, :, 0]
    n=len(data)
    standardize(data)
    k_vals.append(len(data[0]))
    dist = []
    for k in k_vals:
        #print("k= "+str(k))
        l2 = 0
        U, S = PCA(data, k)
        V = np.matmul(data, np.matrix.transpose(U))
        xPrime= mat.transpose(np.matmul(np.transpose(U), np.transpose(V)))
        for i in range(5):
            rand_number = np.random.randint(0, n - 1)
            #plot_vectors(np.array([data[rand_number], xPrime[rand_number]]), h, w, 1, 2)
            l2 += lin.norm(data[rand_number] - xPrime[rand_number])
        dist.append(l2)
    plt.plot(k_vals, dist, color='r', marker='o')
    plt.xlabel('dim')
    plt.ylabel('L2 norms')
    plt.title('L2_distances after changing dimensions')
    plt.show()


def section_d():
        lfw_people = load_data()
        names = lfw_people.target_names
        data, Y_values, h, w = get_pictures_by_names(names)
        data = np.array(data)[:, :, 0]
        standardize(data)
        X_train, X_test, Y_train, Y_test = train_test_split(data, Y_values, test_size=0.25, random_state=0)
        k_vals = [1, 5, 10, 30, 50, 100, 150, 300, len(data[0])]
        scores = []
        for k in k_vals:
                print("k is {}".format(k))
                U, S = PCA(X_train, k)
                A = np.matmul(X_train, mat.transpose(U))
                X_test_newDim = mat.transpose(np.matmul(U, np.matrix.transpose(X_test)))
                res = svm.SVC(kernel='rbf', C=1000, gamma=10 ** -7).fit(A, Y_train)
                scores.append(res.score(X_test_newDim, Y_test))
        plt.plot(k_vals, scores, color='blue')
        plt.xlabel('k values')
        plt.ylabel('Accuracy')
        plt.title('Accuracy through different dimensions')
        plt.show()


def get_pictures_by_names(names):
    """
    Given a name returns all the pictures of the person with this specific name.
    YOU CAN CHANGE THIS FUNCTION!
    THIS IS JUST AN EXAMPLE, FEEL FREE TO CHANGE IT!
    """
    lfw_people = load_data()
    images_res = []
    target_labels = []
    Y_vals = []
    samples, h, w = lfw_people.images.shape
    names_lst=list(lfw_people.target_names)
    for i in range(len(names)):
        target_labels.append(names_lst.index(names[i]))
    for image, target in zip(lfw_people.images, lfw_people.target):
        if target in target_labels:
            imageVec = image.reshape((h * w, 1))
            images_res.append(imageVec)
            Y_vals.append(names_lst[target])
    return images_res, Y_vals, h, w


def plot_vectors(U, h, w, row, col):
    plt.figure(figsize=(1.5 * col, 2. * row))
    plt.subplots_adjust(0.6, 0.5, 1.5, 1.5)
    n= U.shape[0]
    for i in range(n):
        plt.subplot(row, col, i + 1)
        plt.imshow(U[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.yticks(())
        plt.xticks(())
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
        #section_c()
        #section_b()
        section_d()
     
