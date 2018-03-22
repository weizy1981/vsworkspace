import pandas as pd
from sklearn.datasets import load_iris
import seaborn as sns
from matplotlib import pyplot as plt

if __name__ == '__main__':
    dataset = load_iris()
    data = dataset['data']
    
    names = ['separ-length', 'separ-width', 'petal-lenght', 'petal-width']
    df = pd.DataFrame(data, columns=names)
    cottMatt = df.corr()

    sns.heatmap(cottMatt, vmin=-1, vmax=1)
    plt.show()
    plt.close()
   