import numpy as np

filename = 'iris.data.csv'

def exchange_class(s):

    s  = s.replace('Iris-setosa', '0').replace('Iris-versicolor', '1').replace('Iris-virginica', '2')
    return s

if __name__ == '__main__':
    data = []

    with open(filename, 'rt') as raw_data:
        lines = raw_data.readlines()
        
        for line in lines:
            line = exchange_class(line.strip('\n'))
            print(line)
            items = line.split(',')
            if len(items) == 5:
                data.append(items)

    dataset = np.array(data, dtype='float')
    dataset = dataset[:, 0:4]
    print(dataset)

    print(dataset.shape)
    print(dataset.mean())
    print(dataset.std())


            