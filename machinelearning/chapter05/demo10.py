from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

if __name__ == '__main__':
    
    dataset = load_iris()
    data = dataset['data']
    target = dataset['target']

    data_train, test_train, target_train, target_test = train_test_split(data, target, 
        test_size=0.3, random_state=7)

    cart = DecisionTreeClassifier()
    model = BaggingClassifier(base_estimator=cart, n_estimators=50, random_state=7)
    model.fit(data_train, target_train)
    target_predict = model.predict(test_train)
    matrix = classification_report(target_test, target_predict)
    print(matrix)