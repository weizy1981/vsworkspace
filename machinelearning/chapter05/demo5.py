from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

if __name__ == '__main__':
    
    dataset = load_boston()
    data = dataset['data']
    target = dataset['target']

    model = LinearRegression()
    kfold = KFold(n_splits=10, random_state=7)
    scoring = 'neg_mean_absolute_error'
    result = cross_val_score(model, data, target, cv=kfold, scoring=scoring)
    print('%.3f(%.3f)' % (result.mean(), result.std()))