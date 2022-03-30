from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

iris=datasets.load_iris()

# print(iris['data'])

x=iris['data'][:,3:]
y=(iris['target']==2).astype(np.int32)

# print(x)
# print(y)


# Train a logistic regression classifier

clf=LogisticRegression()
clf.fit(x,y)

example=clf.predict(([[4.5]]))

print(example)

# Using matplotlib to plot the visualization
X_new = np.linspace(0,3,1000).reshape(-1,1)
y_prob = clf.predict_proba(X_new)
# print(X_new)
print(y_prob[:1])

plt.plot(X_new, y_prob[:,1], "g-", label="virginica")
plt.show()
