from sklearn import tree
from binary_classification import load_data, train, predict, accuracy

# A decision tree works by splitting the data into branches based on feature thresholds, making sequential decisions until it 
# reaches a leaf node that assigns a class label. The model recursively selects the feature and split point that best separates 
# the classes, using metrics like Gini impurity or information gain to optimize each decision. I chose a decision tree because it 
# felt relatively easily interpretable. 
 
# linear regression model setup
X_train, X_test, y_train, y_test, feature_names = load_data()

w, b, losses = train(X_train, y_train, alpha=0.01, n_epochs=100)

train_pred = predict(X_train, w, b)
test_pred = predict(X_test, w, b)
    
train_acc = accuracy(y_train, train_pred)
test_acc = accuracy(y_test, test_pred)

# Decision Tree Implementation
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

tree_test_predict = clf.predict(X_test)
tree_test_acc = accuracy(y_test, tree_test_predict)

print(f"Linear Regression Model Accuracy: {test_acc}")
print(f"Decsion Tree Accuracy: {tree_test_acc}")

# The linear regression model performed better. Its high performance could be attributed to 
# the features have a clean, linear boundary distinguishing malignant from benign tumors. 
# The decision tree may not have been configured with the right depth to best fit this dataset.


