from sklearn.linear_model import Ridge
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

if __name__ == '__main__':
    
    dataset = load_boston()
    data = dataset['data']
    target = dataset['target']

    base_model = Ridge()
    params = {'alpha': uniform()}
    kfold = KFold(n_splits=10, random_state=7)
    model = RandomizedSearchCV(estimator=base_model, param_distributions=params, cv=kfold)
    result = model.fit(data, target)
    print('Best Score: %.3f, Best Param: %s' % (result.best_score_, result.best_params_))
    
