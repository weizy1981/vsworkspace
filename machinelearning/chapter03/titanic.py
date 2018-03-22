import pandas as pd
from pandas import read_csv
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import os

if __name__ == '__main__':
    filename = 'titanic_train.csv'
    path = os.path.dirname(__file__)
    filepath = os.path.join(path, filename)
    
    dataset = read_csv(filepath, index_col=['PassengerId'])
    #print(dataset.head())

    target = dataset['Survived']
    data = pd.DataFrame(dataset.drop(columns=['Survived', 'Name', 'Ticket']))
    #print(data.head())

    # 观察数据，判读是否有缺失值
    #print(data.info())

    data['Age'].fillna(int(data['Age'].mean()), inplace=True)
    data['Cabin'].fillna(0, inplace=True)
    data['Cabin'] = data['Cabin'].apply(lambda x: 0 if x == 0 else 1)
    #print(data.groupby(by=['Embarked']).count())
    data['Embarked'].fillna('S', inplace=True)
    #print(data.info())

    # 观察数据
    print(data.head(10))
    print(data.describe())

    # 图表展示数据
    sns.pairplot(data)
    plt.show()
    plt.close()
    print(data[['Age', 'Fare']])
    
    corrMatt = data.corr()
    sns.heatmap(corrMatt)
    plt.show()
    plt.close()
