# data
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
# plot
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# train
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV  
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score 
# decision
from itertools import product
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
#from sklearn.inspection import DecisionBoundaryDisplay

def get_random_data(N):
    mu_1, mu_2, mu_3 = 1, 2, 3
    sigma_1, sigma_2, sigma_3 = 100, 100, 100
    mean = torch.Tensor([[mu_1, mu_2, mu_3]])
    mean = torch.Tensor([mu_1, mu_2, mu_3])
    #cov = torch.Tensor([[1, 1, 1], [1, 2, 2], [1, 2, 3]])
    cov = torch.Tensor([[sigma_1**2, 0, 0], [0, sigma_2**2, 0], [0, 0, sigma_3**2]])
    distrib = MultivariateNormal(loc=mean, covariance_matrix=cov)
    #distrib = torch.distributions.normal.Normal(loc=mean, scale=torch.diagonal(cov))

    mask_prob = torch.tensor([1.0, 1.0, 0.1])
    mask = torch.distributions.bernoulli.Bernoulli(probs=mask_prob, logits=None)

    x = distrib.sample(sample_shape = (N,))

    y1 = (x[:,0] >= mu_1).int()

    x1 = (x[:,1] - mu_2) / sigma_2
    x2 = (x[:,2] - mu_3) / sigma_3
    y2 = (x1**2 + x2**2 <= 1.).int()

    #y2 = (x[:,1] >= mu_2).int()

    #y = (y1*y2)#.remainder(2)
    #y = y2#.remainder(2)
    y = (y1+2*y2)

    c1=(y==0).sum()
    c2=(y==1).sum()
    c3=(y==2).sum()
    c4=(y==3).sum()

    print("================================")
    print("c1, c2, c3, c4 = ", c1, c2, c3, c4)
    print("================================")

    x = x * mask.sample(sample_shape = (N,))

    #return x[:,:,0], y
    return x, y


def plot(x,  y, n_features, n_class, C_max = 2) :

    n1 = 0
    n2 = 0
    n = 0
    for f1 in range(n_features) :
        for f2 in range(f1+1, n_features) :
            n1+=1
            for f3 in range(f2+1, n_features) :
                n2+=1
    
    lb = n1+n2
    if lb <= C_max : L, C = 1, lb
    else : 
        C = C_max
        L = lb // C + lb % C

    figsize = (6, 4)
    figsize = (6, 4*5)
    figsize = (15, 8)
    figsize = (15, 10)
    figsize=(C*figsize[0], L*figsize[1])

    fig = plt.figure(figsize=figsize)


    n_class = 4
    classes = range(n_class)

    x_cs = []
    #xs, ys, zs = [], [], []

    #features = [[]]*n_features
    features = [[] for _ in range(n_features)]

    for i in classes :
        xc = x[y == i]
        x_cs.append(xc)
        for j in range(n_features) :
            features[j].append(xc[:,j])

    
    colors = ['r', 'b', 'y', 'g']
    markers = ['o', '^', '*', '+']

    k = 1
    for f1 in range(n_features) :
        for f2 in range(f1+1, n_features) :
            n1+=1
            for f3 in range(f2+1, n_features) :
                ax = fig.add_subplot(L, C, k, projection='3d')
                k+=1
                for i in classes :
                    ax.scatter(features[f1][i], features[f2][i], features[f3][i], c=colors[i], marker=markers[i])
                    ax.set_xlabel(f'x{f1}')
                    ax.set_ylabel(f'x{f2}')
                    ax.set_zlabel(f'x{f3}')

    for f1 in range(n_features) :
        for f2 in range(f1+1, n_features) :
            ax = fig.add_subplot(L, C, k)
            k+=1
            for i in classes :
                ax.scatter(features[f1][i], features[f2][i], c=colors[i], marker=markers[i], label=i)
            ax.set_xlabel(f'x{f1}')
            ax.set_ylabel(f'x{f2}')
            ax.legend()

    plt.show()

#######################
def print_evaluation_scores(y, predicted):
    print("accuracy_score : ", accuracy_score(y, predicted))
    print("f1_score : ", f1_score(y, predicted, average="macro"))
    print("recall_score : ", recall_score(y, predicted, average="macro"))
    
###################
###################
N = 1000
x, y = get_random_data(N)
x = x.round()

plot(x, y, n_features=3, n_class=4)

########
train_pct=10
seed = 0
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1-train_pct/100, random_state = seed)

############ classifier ################

classifiers = {}

##############
# parameters = {'C': np.linspace(start = 0.0001, stop= 100, num=100)}
# grid_search = GridSearchCV(LogisticRegression(), parameters, n_jobs = -1)
# grid_search.fit(x_train, y_train)

# print('best parameters mybag: ', grid_search.best_params_)
# print('best scrores mybag: ', grid_search.best_score_)

# C=grid_search.best_params_['C']
C=30.3031
classifiers['log_reg'] = LogisticRegression(penalty = "l2", C = C, solver = "newton-cg", random_state = 0, n_jobs = -1)


##############
classifiers["svc_2vsrest"] = OneVsRestClassifier(LinearSVC(random_state=0))

#############
classifiers["gnb"] = GaussianNB()

#############
classifiers["dt"] = DecisionTreeClassifier(max_depth=4)
classifiers["knn"] = KNeighborsClassifier(n_neighbors=7)
classifiers["svc"] = SVC(gamma=0.1, kernel="rbf", probability=True)

# classifiers["eclf"] = VotingClassifier(
#     estimators=[(k, v) for k, v in classifiers.items() if k!="svc_2vsrest"],
#     voting="soft",
#     weights=[2, 1, 2, 2, 1],
# )

#############
for k in classifiers :
    classifiers[k] = classifiers[k].fit(x_train, y_train)


#############
y_pred = {}
for k in classifiers :
    y_test_predicted = classifiers[k].predict(x_test)
    try : y_test_predicted_scores = classifiers[k].decision_function(x_test)
    except AttributeError : pass
    print(f'===== {k}_train : ', classifiers[k].score(x_train, y_train))
    print(f'===== {k}_test : ', classifiers[k].score(x_test, y_test))
    print_evaluation_scores(y_test, y_test_predicted)

    y_pred[k] = y_test_predicted

# for k in classifiers :
#     plot(x_test, y_pred[k], n_features=3, n_class=4)


# # Plotting decision regions
# f, axarr = plt.subplots(3, 2, sharex="col", sharey="row", figsize=(10, 8))
# for idx, clf, tt in zip(
#     product([0, 1], [0, 1]),
#     classifiers.values(),
#     classifiers.keys(),
# ):
#     X = x#_test
#     Y = y#_test
#     DecisionBoundaryDisplay.from_estimator(
#         clf, X, alpha=0.4, ax=axarr[idx[0], idx[1]], response_method="predict"
#     )
#     axarr[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=Y, s=20, edgecolor="k")
#     axarr[idx[0], idx[1]].set_title(tt)

# plt.show()

X = x#_test
Y = y#_test

#plot_decision_boundary(X, Y, None, steps=1000, color_map='Paired')