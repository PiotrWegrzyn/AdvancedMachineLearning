import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


data = pd.read_csv('banknotes.txt', sep=",")

print(data.head())


y = data['class']
X = data.drop('class', axis=1)

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y)

hidden_layers = [1, 2, 3]
neurons = [50, 100, 150]
hidden_layer_sizes = [tuple(n_neurons for _ in range(n_layers)) for n_neurons in neurons for n_layers in hidden_layers]

hyperparameters = [{
    'hidden_layer_sizes': hidden_layer_sizes,
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'max_iter': [500]
}]

clf = GridSearchCV(MLPClassifier(), hyperparameters, cv=5, verbose=True)
clf.fit(X_train, y_train)

best_model = clf.best_estimator_

print(best_model)
best_model.fit(X_train, y_train)

y_predicted = best_model.predict(X_test)
print(confusion_matrix(y_test, y_predicted))

print(classification_report(y_test, y_predicted))

