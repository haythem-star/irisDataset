# from itertools import cycle
# import numpy as np
#
from sklearn import datasets  # import datasets from sklearn library
# irisData = datasets.load_iris()
# # print (irisData.data)
# # print (irisData.target)
# # print (irisData.target_names)
#
# import pylab as pl
# def plot_2D(data, target, target_names):
#     colors = cycle('rgbcmykw') # cycle de couleurs
#     target_ids = range(len(target_names))
#     pl.figure()
#     for i, c, label in zip(target_ids, colors, target_names):
#         pl.scatter(data[target == i, 0], data[target == i, 1], c=c, label=label)
#     pl.legend()
#     x = np.linspace(0, 10, 1000)
#     pl.plot(x,0.35*x+1.3)
#     pl.show()
#
# plot_2D(irisData.data,irisData.target,irisData.target_names)

# from sklearn import naive_bayes
# nb = naive_bayes.MultinomialNB(fit_prior=True)# un algo d'apprentissage
irisData = datasets.load_iris() #load iris dataset
# # fit (training) the model with all the iris_Dataset except last instance
# nb.fit(irisData.data[:-1], irisData.target[:-1])
# # try to predict the type of iris for the 32nd element
# p31 = nb.predict([irisData.data[31]])
# print(irisData.target_names[p31])
# # try to predict the type of iris for the last element
# plast = nb.predict([irisData.data[-1]])
# print(irisData.target_names[plast])
# # try to predict the type of iris for all the dataset
# p = nb.predict(irisData.data[:])
# print(p)



# from sklearn import naive_bayes
# nb = naive_bayes.MultinomialNB(fit_prior=True)
# nb.fit(irisData.data[:99], irisData.target[:99])
# p= nb.predict(irisData.data[100:149])
#
# print(p)

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
dt = DecisionTreeClassifier()
score= cross_val_score(dt,irisData.data,irisData.target,cv=10)
print(score)
print("Accuracy = {}".format(score.mean()))


# data_train,data_test,target_train,target_test = \
#     train_test_split(irisData.data,irisData.target,random_state=5)
# nb.fit(data_train,target_train)
# p = nb.predict(data_test)
# print("real type of iris:")
# print(target_test)
# print("prediction :")
# print(p)
# print("score of model = {}".format(nb.score(data_test,target_test)))



# ea = 0
# for i in range(len(data_test)):
#     if (p[i] != target_test[i]):
#         ea = ea+1
# print(ea/len(data_test))
#
# ae = 1 - (((target_test - p)==0).sum())/len(data_test)
#
# print(ae)






