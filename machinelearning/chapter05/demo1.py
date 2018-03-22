from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

if __name__ == '__main__':
    
    dataset = load_iris()
    data = dataset['data']
    target = dataset['target']

    model = LogisticRegression()
    kfold = KFold(n_splits=10, random_state=7)
    scoring = 'accuracy'
    result = cross_val_score(model, data, target, cv=kfold, scoring=scoring)
    print('%.3f(%.3f)' % (result.mean(), result.std()))