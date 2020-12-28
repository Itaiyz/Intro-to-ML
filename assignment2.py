#################################
# Your name: Itai Zemah
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals
import random


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        x_vals= np.random.uniform(0,1, size=m)
        x_vals.sort()
        y_vals= []
        for i in range(m):
            x=x_vals[i]
            y=random.uniform(1,100)
            if 0<=x<=0.2 or 0.4<=x<=0.6 or 0.8<=x<=1:
                if 20<y<=100:
                    y=1
                else:
                    y=0
            else:
                if 10<y<=100:
                    y=0
                else:
                    y=1
            y_vals.append(y)
        res= np.column_stack((x_vals, y_vals))
        #plt.scatter(x_vals,y_vals)
        for val in [0.2,0.4,0.6,0.8]:
            plt.axvline(x=val)
        #plt.show()
        return res

           

    def draw_sample_intervals(self, m, k):
        """
        Plots the data as asked in (a) i ii and iii.
        Input: m - an integer, the size of the data sample.
               k - an integer, the maximum number of intervals.

        Returns: None.
        """
        arr= Assignment2.sample_from_D(self,m)
        plt.ylabel("P")
        plt.xlabel("intervals")
        plt.ylim((-0.1,1.1))
        #ax.yaxis.limit_range_for_scale(-0.1,1.1)
        for val in [0.2,0.4,0.6,0.8]:
            plt.axvline(x=val)
        #plt.plot(arr[0], arr[1], k, '.')
        #plt.show()
        return None
        
        

    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        true_err= []
        emp_err= []
        for m in range(m_first, m_last+1, step):
            m_true_err= np.zeros(T, dtype= float)
            m_emp_err= np.zeros(T, dtype= float)
            for i in range(T):
                vals= self.sample_from_D(m)
                x_vals = vals[:,0]
                y_vals = vals[:,1]
                intervals_lst, eP_s = intervals.find_best_interval(x_vals, y_vals, k)
                m_true_err[i] = self.calcErr(intervals_lst)
                m_emp_err[i] = eP_s/m
            true_err.append(m_true_err.mean())
            emp_err.append(m_emp_err.mean())

        X = [m for m in range(m_first, m_last + 1, step)]
        plt.clf()
        plt.ylim((-0.1,1.1))
        plt.xlabel("step")
        plt.ylabel("error")
        plt.plot( true_err, marker='o', color = 'blue', label = 'true_error')
        plt.plot( emp_err, marker='o',  color = 'red', label = 'empirical_error')
        #plt.legend()
        #plt.show()   
        res = np.column_stack((emp_err, true_err))
        return res

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        true_err= []
        emp_err= []
        
        for k in range(k_first, k_last+1, step):
            #print(k)
            vals= self.sample_from_D(m)
            x_vals = vals[:,0]
            y_vals = vals[:,1]
            intervals_lst, eP_s = intervals.find_best_interval(x_vals, y_vals, k)
            true_err.append(self.calcErr(intervals_lst))
            emp_err.append(eP_s/m)
            
        plt.clf()
        plt.ylim((-0.1,1.1))
        plt.xlabel("step")
        plt.ylabel("error")
        X = [k for k in range(k_first, k_last + 1, step)]
        plt.plot( true_err, marker='o', color = 'blue', label = 'true_error')
        plt.plot( emp_err, marker='o',  color = 'red', label = 'empirical_error')
        plt.legend()
        #plt.show()   
        minimum = np.argmin(emp_err) #index of minimal empirical error
        return minimum*step + k_first

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Runs the experiment in (d).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        true_err= []
        emp_err= []
        penalties=[]
        
        for k in range(k_first, k_last+1, step):
            #print(k)
            vals= self.sample_from_D(m)
            x_vals = vals[:,0]
            y_vals = vals[:,1]
            intervals_lst, eP_s = intervals.find_best_interval(x_vals, y_vals, k)
            true_err.append(self.calcErr(intervals_lst))
            emp_err.append(eP_s/m)
            penalties.append(Assignment2.penalty(self,k,m,0.1))

        plt.clf()
        plt.xlabel("step")
        plt.ylabel("error")
        #X = [k for k in range(k_first, k_last + 1, step)]
        sumPenalEmp=[penal+emp for penal,emp in zip(penalties,emp_err)]
        plt.xlim(0,15)
        plt.plot( true_err, marker='o', color = 'blue', label = 'true_error')
        plt.plot( emp_err, marker='o',  color = 'red', label = 'empirical_error')
        plt.plot(penalties, marker='o', color= 'green', label=' penalty')
        plt.plot(sumPenalEmp, marker='o', color= 'orange', label='emp_error+ penalty')
        plt.legend()
        #plt.show()   
        minimum = np.argmin(emp_err) #index of minimal empirical error
        return minimum*step + k_first

    def cross_validation(self, m, T):
        """Finds a k that gives a good test error.
        Chooses the best hypothesis based on 3 experiments.
        Input: m - an integer, the size of the data sample.
               T - an integer, the number of times the experiment is performed.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        Tk = []  
        Tk_chosen = []   
        values = self.sample_from_D(m)
        
        for i in range(T):
            #Divides the sample into S_ho and S_t. S_ho is consisted of the last 20% of the array
            np.random.shuffle(values) 
            size = int(0.8*m)
            train_values = values[:size]  
            holdout_values = values[size:]   
            train_values = np.array(sorted(train_values, key=lambda x: x[0])) #sorting the array according to the x values and converting to ndarray
            x_train = train_values[:,0]
            y_train = train_values[:,1]
            x_holdout = holdout_values[:,0]
            y_holdout = holdout_values[:,1]
            minK = 1   
            min_Kerr = 1   
            for k in range(1,11): 
                intervals_lst, error_cnt = intervals.find_best_interval(x_train, y_train, k) 
                holdout_err = self.calcEmpErr(intervals_lst, x_holdout, y_holdout)
                if(holdout_err < min_Kerr):   
                    min_Kerr = holdout_err
                    minK = k
            Tk.append(mink)
            Tk_chosen.append(min_Kerr)
            

            
        minimum =  np.argmin(Tk_chosen) 
        return Tk_chosen[min_index]

    #################################
    # Place for additional methods
    def calcErr(self, intervals_list):
         def intersect(intervals_list, A, B):
             """
                 returns the intersection of the intervals with [A, B] in order to calculate the true error
             """
             res=0
             for interval in intervals_list:
                 a = interval[0]
                 b = interval[1]
                 start = max(A,a)
                 end = min(B,b)
                 if(start<end):
                     res += (end-start)
             return res 

         res = 0
         sum1 = intersect(intervals_list,0, 0.2)+ intersect(intervals_list,0.4, 0.6)+ intersect(intervals_list,0.8, 1)
         res += sum1 * 0.2
         res += (0.6-sum1) * 0.8
         sum2 = intersect(intervals_list,0.2, 0.4)+intersect(intervals_list,0.6, 0.8)    
         res += sum2 * 0.9
         res += (0.4-sum2) * 0.1
         return res

    def penalty(self, k ,m, delta):
        """
        calculates penalty for question d
        """
        #  VCdim(H_k) = 2k
        from math import exp,log,sqrt
        logVal = (m*exp(1))/k #|Hk|=k
        den = 2*k*log(logVal) #d=2k
        nom = log(4/delta)
        res = 8/m*(den+nom)
        
        return sqrt(res)
                   
      
    def calcEmpErr(self, intervals_list, x_vals, x_labels):
        """
        return number od 0 labels in the hypothesis 
        """
        error = 0
        n= len(x_vals)
        for i in range(n):
            h_x = 0  
            for val in intervals_list:
                beg = val[0]
                end = val[1]
                if beg<=x_vals[i]<=end:
                    h_x = 1
                    break
                if(x_vals[i]<beg): 
                    break
            if(h_x != x_labels[i]):
                error += 1
        return error/n

    
        


if __name__ == '__main__':
    ass = Assignment2()
    ass.draw_sample_intervals(100, 3)
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500, 3)

