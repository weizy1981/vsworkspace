from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':
    
    dataset = load_iris()
    data = dataset['data']
    target = dataset['target']

    base_model = KNeighborsClassifier()
    params = {'n_neighbors': [1, 3, 5, 7, 9]}
    kfold = KFold(n_splits=10, random_state=7)
    model = GridSearchCV(estimator=base_model, param_grid=params, cv=kfold)
    result = model.fit(data, target)
    print('Best Score: %.3f, Best Param: %s' % (result.best_score_, result.best_params_))
    
