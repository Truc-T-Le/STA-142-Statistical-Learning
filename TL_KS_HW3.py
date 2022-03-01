#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import make_interp_spline, BSpline
from pygam import LogisticGAM, s, f, l
import time
from IPython.display import Image 
from IPython.core.display import HTML 
from numpy import random


# # STA 142A Homework 3 {-}
# Team Members: Truc Le and

# # Problem 1. Smoothing Splines.

# ## Question 2 from page 322

# $\hat{g}=arg\underset{g}{min}(\sum_{i=1}^n(y_i - g(x_i))^2 + \lambda\int[g^{(m)}(x)]^2dx)$

# In[2]:


# generating a random linear plot
np.random.seed(1)
x = [random.random() for i in range(50)]
y = [0.5+0.5*i+0.2 for i in x]
plt.plot(x,y)
plt.show()


# ### a) $\lambda = \infty$ , $m = 0$
# * As $\lambda \rightarrow \infty$ the penality term has now become paramount, causing g(x) $\rightarrow$ 0.
#     * $\hat{g}$ = 0
# * for m = 0:
#     * $\hat{g}=arg\underset{g}{min}(\sum_{i=1}^n(y_i - g(x_i))^2 + \lambda\int[g^{(0)}(x)]^2dx)$  $\rightarrow$ $\hat{g}=arg\underset{g}{min}(\sum_{i=1}^n(y_i - g(x_i))^2 + \lambda\int[g(x)]^2dx)$ 

# In[3]:


np.random.seed(1)
x = [random.random() for i in range(50)]
y = [5*i-2 for i in x]
plt.plot(x,y)
plt.axhline(y=0, color='r', linestyle='-')
plt.title('lambda= inf, m=0')
plt.xlabel('x')
plt.ylabel('y') 
plt.show()


# ### b) $\lambda = \infty$ , $m = 1$
# * As $\lambda \rightarrow \infty$ the penality term has now become paramount, causing g'(x) $\rightarrow$ 0.
#     * $\hat{g} = c = \frac{1}{n}\sum_{i=1}^n y_i$
# * for m = 1:
#     * $\hat{g}=arg\underset{g}{min}(\sum_{i=1}^n(y_i - g(x_i))^2 + \lambda\int[g^{(1)}(x)]^2dx)$  $\rightarrow$ $\hat{g}=arg\underset{g}{min}(\sum_{i=1}^n(y_i - g(x_i))^2 + \lambda\int[g'(x)]^2dx)$ 

# In[4]:


np.random.seed(1)
x = [random.random() for i in range(50)]
y = [5*i-2 for i in x]
plt.plot(x,y)
plt.axhline(y=np.mean(y), color='r', linestyle='-')
plt.title('lambda= inf, m=1')
plt.xlabel('x')
plt.ylabel('y') 
plt.show()


# ### c) $\lambda = \infty $ , $m = 2$
# * As $\lambda \rightarrow \infty$ the penality term has now become paramount, causing g''(x) $\rightarrow$ 0.
#     * $\hat{g} = ax+b$
# * for m = 2:
#     * $\hat{g}=arg\underset{g}{min}(\sum_{i=1}^n(y_i - g(x_i))^2 + \lambda\int[g^{(2)}(x)]^2dx)$  $\rightarrow$ $\hat{g}=arg\underset{g}{min}(\sum_{i=1}^n(y_i - g(x_i))^2 + \lambda\int[g''(x)]^2dx)$ 

# In[5]:


np.random.seed(1)
x = [random.random() for i in range(50)]
y = [5*i-2 for i in x]
plt.plot(x,y)
x2 = np.linspace(min(x),max(x), 50)
f = interp1d(x, y, kind='linear')
y2 = f(x2)
plt.plot(x2,y2)
plt.title('lambda= inf, m=2')
plt.xlabel('x')
plt.ylabel('y') 
plt.show()


# ### d) $\lambda = \infty $ , $m = 3$
# * As $\lambda \rightarrow \infty$ the penality term has now become paramount, causing g'''(x) $\rightarrow$ 0.
#     * $\hat{g} = ax^2+bx+c$
# * for m = 3:
#     * $\hat{g}=arg\underset{g}{min}(\sum_{i=1}^n(y_i - g(x_i))^2 + \lambda\int[g^{(3)}(x)]^2dx)$  $\rightarrow$ $\hat{g}=arg\underset{g}{min}(\sum_{i=1}^n(y_i - g(x_i))^2 + \lambda\int[g'''(x)]^2dx)$ 

# In[6]:


np.random.seed(1)
x = [random.random() for i in range(50)]
y = [5*i-2 for i in x]
plt.plot(x,y)
x2 = np.linspace(min(x),max(x), 50)
f = interp1d(x, y, kind='quadratic')
y2 = f(x2)
plt.plot(x2,y2)
plt.title('lambda= inf, m=3')
plt.xlabel('x')
plt.ylabel('y') 
plt.show()


# ### e) $\lambda = 0 $ , $m = 3$
# * As $\lambda = 0 $ the penality term no longer play any role, so g now becomes an interpolating spline.
# * for m = 3:
#     * $\hat{g}=arg\underset{g}{min}(\sum_{i=1}^n(y_i - g(x_i))^2 + \lambda\int[g^{(3)}(x)]^2dx)$  $\rightarrow$ $\hat{g}=arg\underset{g}{min}(\sum_{i=1}^n(y_i - g(x_i))^2 + \lambda\int[g'''(x)]^2dx)$ 

# In[7]:


np.random.seed(1)
x = [random.random() for i in range(50)]
y = [5*i-2 for i in x]
plt.plot(x,y)
x2 = np.linspace(min(x),max(x), 50)
f = interp1d(x, y, kind='previous')
y2 = f(x2)
plt.plot(x2,y2)
plt.title('lambda= 0, m=3')
plt.xlabel('x')
plt.ylabel('y') 
plt.show()


# ## Question 5 from page 323. 

# $\hat{g}_1=arg\underset{g}{min}(\sum_{i=1}^n(y_i - g(x_i))^2 + \lambda\int[g^{(3)}(x)]^2dx)$ 
# 
# $\hat{g}_2=arg\underset{g}{min}(\sum_{i=1}^n(y_i - g(x_i))^2 + \lambda\int[g^{(4)}(x)]^2dx)$ 

# ### a) As $\lambda \rightarrow \infty$, will $\hat{g}_1$ or $\hat{g}_2$ have the smaller training RSS?
# 
# * $\hat{g}_2$ will likely have a smaller training RSS since it has a higher polynomial order, therefore, making the curve more flexible to fit the training data set. 

# ### b) As $\lambda \rightarrow \infty$, will $\hat{g}_1$ or $\hat{g}_2$ have the smaller test RSS?
# 
# * We cannot give a definite answer to which curve will hae a smaller test RSS, since we do not know the true relationship between the x and y valuables. The $\hat{g}_2$ extra fexiblity could cause it to overfit and have a larger tet RSS, however, if extra complexity is necessary then $\hat{g}_2$ will result in a smaller test RSS. In the otherhand, $\hat{g}_1$ could underfit if the extra complexity is required, resulting in $\hat{g}_1$ having a larger test RSS value.

# ### c) For $\lambda = 0$, will $\hat{g}_1$ or $\hat{g}_2$ have the smaller training and test RSS?
# 
# * if $\lambda = 0$ then both $\hat{g}_1$ and $\hat{g}_2$ will be equal to each other, therefore, they will have the same training and test RSS.

# # Problem 2. Trees and Bagging

# ## Question 1 from page 361

# In[8]:


PATH = "/Users/trucle/Desktop/Prob2.jpg"
Image(filename = PATH , width=1000, height=1000)


# ## Question 5 from page 362

# P(Class is Red|X): 0.1, 0.15, 0.2, 0.2, 0.55, 0.6, 0.6, 0.65, 0.7, 0.75

# * Majority vote:
#     * 0.1 $\rightarrow$ Green because (P(Green)=0.9, P(Red)=0.1)
#     * 0.15 $\rightarrow$ Green because (P(Green)=0.85, P(Red)=0.15)
#     * 0.2 $\rightarrow$ Green because (P(Green)=0.8, P(Red)=0.2)
#     * 0.2 $\rightarrow$ Green because (P(Green)=0.8, P(Red)=0.2)
#     * 0.55 $\rightarrow$ Red because (P(Green)=0.45, P(Red)=0.55)
#     * 0.6 $\rightarrow$ Red because (P(Green)=0.4, P(Red)=0.6)
#     * 0.6 $\rightarrow$ Red because (P(Green)=0.4, P(Red)=0.6)
#     * 0.65 $\rightarrow$ Red because (P(Green)=0.35, P(Red)=0.65)
#     * 0.7 $\rightarrow$ Red because (P(Green)=0.3, P(Red)=0.7)
#     * 0.75 $\rightarrow$ Red because (P(Green)=0.25, P(Red)=0.75)
# 
# Under the majority vote, Red would get 6 votes and Green would get 4 votes. Therefore, the final classification for X would be Red 

# * Average Probability Approach
#     * Average = $\frac{0.1 + 0.15 + 0.2 + 0.2 + 0.55 + 0.6 + 0.6 + 0.65 + 0.7 + 0.75}{10}$ = 0.45
#  
# The average probability of the 10 probabilities is 0.45, meaning that P(Red) = 0.45 and P(Green) = 0.55. Therefore, under the average probability approach, we would classify X as Green.

# # Problem<div class="cite2c-biblio"></div> 3. GAM+Splines

# For even j, the $j^{th}$-coordinate of X is distributed as
# * $X_j|(Y=1)$ is a t-distribution with 1 degree of freedom with mean 2
# * $X_j|(Y=0)$ is a t-distribution with 1 degree of freedom with mean 0 
# 
# For odd j, the $j^{th}$-coordinate of X is distributed as
# * $X_j|(Y=1)$ is an exponential distribution with $\lambda = 1$
# * $X_j|(Y=1)$ is an exponential distribution with $\lambda = 3$ 
# 
# 
# P(Y=1) = 0.5

# ## a) p = 10

# In[9]:


# n = size of matrix
# p = the given P value
# this function is to generate samples
def data(n,p, lam1, lam2, mu):
    # calculating binomial distribution for odd index
    # n1,n2 = odd
    # n3,n4 = even
    n1,n3 = np.random.binomial(n/2,0.5), np.random.binomial(n/2,0.5) 
    n2,n4 = int(n/2 - n1), int(n/2 - n3)

    # function is to return a list of binary outputs
    def repeat(numA,numB):
        y  = [1]*numA
        y_0 = [0]*numB

        for i in y_0:
            y.append(i)

        return y

    y_e = repeat(n1,n2)
    y_o = repeat(n3,n4)

    # alculate the conditional distribution for odd and even depending on the binary
    x_e, x_o = [], []
    for i in range(n1):
        x_o.append(np.random.exponential(1/lam1,p))
    for j in range(n2):
        x_o.append(np.random.exponential(1/lam2,p))
    for i in range(n3):
        x_e.append((np.random.standard_t(1,p) + mu))
    for j in range(n4):
        x_e.append(np.random.standard_t(1,p))

    #concating the even and odd lists into on big list in alternating fashion
    #so elements in odd list are placed into odd index value and vice versa
    def concat(x_e, x_o, y_e, y_o,n):
        X, Y =[None]*n,[None]*n

        # even
        X[::2] = x_e
        Y[::2] = y_e
        #odd
        X[1::2] = x_o
        Y[1::2] = y_o
        
        return X,Y
    return concat(x_e, x_o, y_e, y_o,n)


def simulation(n,p,lam1,lam2,mu):
    test_error = []
    for i in range(n):
        # Generate the training and testing data
        x_train, y_train = data(n,p,lam1,lam2,mu)
        x_test, y_test = data(n,p,lam1,lam2,mu)
        
        # fitting the model
        logGam = LogisticGAM()
        logGam.fit(x_train, y_train)
        
        y_pred = logGam.predict(x_test)
        err = sum(y_pred != y_test)/n
        test_error.append(err)
        # predict the y using the trained model
        # calculate the test error
#         y_pred = logGam.predict(x_test)
#         err = sum(y_pred != y_test)/n
#         test_error.append(err)

    plt.figure(num = None, figsize=(8,6), dpi = 80, facecolor="w", edgecolor="k")
    plt.boxplot(test_error, labels=["P = {}".format(p)])
    plt.text(1.09, 0.22, "Mean: {:.2f}".format(np.mean(test_error)))
    plt.text(1.09, 0.2, "Var: {:.5f}".format(np.var(test_error)))
    plt.title("Multivariate Logistic Regression: P = {}".format(p))
    plt.ylabel("Test Error")
    plt.show()


    
    return test_error
start_time = time.time()
simulation(100,10,1,3,2)    
print("--- %s seconds ---" % (time.time() - start_time))


# ## b) p = 30

# In[ ]:


start_time = time.time()
p30 = simulation(100,30,1,3,2)    
print("--- %s seconds ---" % (time.time() - start_time))


# As $x_i$ goes from $x_i \in \mathbb{R}^{10}$ to $x_i \in \mathbb{R}^{30}$, the computation times increases almost 10 fold, from 20 seconds to 204 seconds. In addition, as P increases, the values for the test error become smaller as well. As expected, we observed that the mean and variance for the test error for P = 30 to be smaller than the mean and variance for P = 10. We would expects overfitting the model to be less of an issue as the sample size becomes larger, thus reducing the test error values as well.

# \textbf{Pledge:}
# \hspace{0.3in}Please sign below (print full name) after checking ($\checkmark$)  the following. If you can not honestly check each of these responses, please email me at kbala@ucdavis.edu to explain your situation.
# \begin{itemize}
# \item We pledge that we are honest students with academic integrity and we have not cheated on this homework.
# 
# \item These answers are our own work.
# 
# \item We did not give any other students assistance on this homework.
# 
# \item We understand that to submit work that is not our own and pretend that it is our is a violation of the UC Davis code of conduct and will be reported to Student Judicial Affairs.
# 
# \item We understand that suspected misconduct on this homework will be reported to the Office of Student Support and Judicial Affairs and, if established, will result in disciplinary sanctions up through Dismissal from the University and a grade penalty up to a grade of 'F' for the course.
# \end{itemize}
# Team Member 1: Truc Le
# \hspace{2in}
# Team Member 2: Alexander Chernikov<div class="cite2c-biblio"></div>
# 
# 
# 
