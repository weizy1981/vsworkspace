from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import os

if __name__ == '__main__':
    # 导入数据
    filename = 'pima_data.csv'
    path = os.path.dirname(__file__)
    filepath = os.path.join(path, filename)

    names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
    data = read_csv(filepath, names=names)
    # 将数据分为输入数据和输出结果
    array = data.values
    data = array[:, 0:8]
    target = array[:, 8]
    num_folds = 10
    seed = 7
    kfold = KFold(n_splits=num_folds, random_state=seed)
    model = LogisticRegression()
    scoring = 'neg_log_loss'
    result = cross_val_score(model, data, target, cv=kfold, scoring=scoring)
    print('Logloss %.3f (%.3f)' % (result.mean(), result.std()))