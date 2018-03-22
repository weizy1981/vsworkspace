from pandas import read_csv
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import numpy as np

seed = 7
n_splits = 10
test_size = 0.3
filename = 'bank.csv'

if __name__ == '__main__':
    np.random.seed(seed=seed)
    
    path = os.path.dirname(__file__)
    filepath = os.path.join(path, filename)
    dataset = read_csv(filepath, delimiter=';')
    print(dataset.info())

    # 将Label转换成数字，便于model处理
    encoder = LabelEncoder()
    dataset['job'] = encoder.fit_transform(dataset['job'])
    dataset['marital'] = encoder.fit_transform(dataset['marital'])
    dataset['education'] = encoder.fit_transform(dataset['education'])
    dataset['default'] = encoder.fit_transform(dataset['default'])
    dataset['housing'] = encoder.fit_transform(dataset['housing'])
    dataset['loan'] = encoder.fit_transform(dataset['loan'])
    dataset['contact'] = encoder.fit_transform(dataset['contact'])
    dataset['month'] = encoder.fit_transform(dataset['month'])
    dataset['poutcome'] = encoder.fit_transform(dataset['poutcome'])
    dataset['y'] = encoder.fit_transform(dataset['y'])
    print(dataset.head())

    data = dataset.values[:, 0: 16].astype('float')
    target = dataset.values[:, 16]
    # 将数据分为训练集和评估集
    data_train, data_test, target_train, target_test = train_test_split(data, target, 
        test_size=test_size, random_state=seed)

    # 对数据进行正态化，并生成模型
    pipelines = {}
    pipelines['RF'] = Pipeline([('scaler', StandardScaler()), ('rf', BaggingClassifier(n_estimators=30))])
    pipelines['ET'] = Pipeline([('scaler', StandardScaler()), ('et', ExtraTreesClassifier(n_estimators=30))])
    pipelines['AB'] = Pipeline([('scaler', StandardScaler()), ('ab', AdaBoostClassifier(n_estimators=30))])

    # 评估模型
    results = []
    for model in pipelines:
        kfold = KFold(n_splits=n_splits, random_state=seed)
        result = cross_val_score(pipelines[model], data_train, target_train, cv=kfold)
        results.append(result)
        print('%s: %.3f(%.3f)' % (model, result.mean(), result.std()))

    # 图表显示评估模型的结果，帮助选择算法
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(pipelines.keys())
    plt.show()

    # 对选定的算法，采用新数据生成评估矩阵
    model = Pipeline([('scaler', StandardScaler()), ('rf', BaggingClassifier(n_estimators=30))])
    model.fit(data_train, target_train)
    target_predict = model.predict(data_test)
    result = classification_report(target_test, target_predict)
    print(result)
    