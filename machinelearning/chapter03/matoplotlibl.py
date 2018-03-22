import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from pandas.plotting import scatter_matrix

if __name__ == '__main__':
    dataset = load_iris()
    data = dataset['data']
    
    names = ['separ-length', 'separ-width', 'petal-lenght', 'petal-width']
    df = pd.DataFrame(data, columns=names)

    scatter_matrix(df)
    plt.show()

    
    