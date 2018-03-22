from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.externals.joblib import dump, load

if __name__ == '__main__':

    dataset = load_iris()
    data = dataset['data']
    target = dataset['target']
    data_train, data_test, target_train, target_test = train_test_split(
        data, target, test_size=0.3, random_state=7)

    model = GaussianNB()
    model.fit(data_train, target_train)
    target_predict = model.predict(data_test)
    result = classification_report(target_test, target_predict)
    print(result)

    # 保存模型
    model_file = 'model.sav'
    with open(model_file, 'wb') as file:
        dump(model, file)
    print()

    # 导入模型
    with open(model_file, 'rb') as file:
        loaded_model = load(file)
        target_predict = loaded_model.predict(data_test)
        result = classification_report(target_test, target_predict)
        print(result)
