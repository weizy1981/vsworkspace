from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from matplotlib import pyplot as plt
import os

if __name__ == '__main__':

    train_path = '20news-bydate-train'
    test_path = '20news-bydate-test'

    path = os.path.dirname(__file__)
    train_path = os.path.join(path, train_path)
    test_path = os.path.join(path, test_path)

    # 导入数据
    dataset_train = load_files(train_path)
    dataset_test = load_files(test_path)

    # 计算词频
    count_vect = CountVectorizer(stop_words='english', decode_error='ignore')
    data_train_count = count_vect.fit_transform(dataset_train.data)
    print(data_train_count.shape)

    # 计算Tfid
    tfid_vect = TfidfVectorizer(stop_words='english', decode_error='ignore')
    data_train_tfid = tfid_vect.fit_transform(dataset_train.data)
    print(data_train_tfid.shape)

    # 定义模型
    models = {}
    models['LR'] = LogisticRegression()
    models['MNB'] = MultinomialNB()
    models['RT'] = BaggingClassifier()

    # 比较算法
    results = []
    for key in models:
        kflod = KFold(n_splits=10, random_state=7)
        result = cross_val_score(models[key], data_train_tfid, dataset_train.target, cv=kflod)
        results.append(result)
        print('%s: %.3f(%.3f)' % (key, result.mean(), result.std()))

    # 图表显示评估模型的结果，帮助选择算法
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(models.keys())
    plt.show()

    # 调参
    param_grid = {}
    param_grid['C'] = [0.1, 3, 5, 13]
    kflod = KFold(n_splits=10, random_state=13)
    model = LogisticRegression()
    search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kflod)
    search_result = search.fit(data_train_tfid, dataset_train.target)
    print('Best Score: %.3f, Best Params: %s' % (search_result.best_score_, search_result.best_params_))

    # 最终模型
    model = LogisticRegression(C=13)
    model.fit(data_train_tfid, dataset_train.target)

    # 使用评估数据集评估模型
    data_test_tfid = tfid_vect.transform(dataset_test.data)
    target_predict = model.predict(data_test_tfid)
    report_result = classification_report(dataset_test.target, target_predict)
    print(report_result)
