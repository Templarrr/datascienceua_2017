def test_requests():
    import requests
    data = requests.get('http://ya.ru')
    assert data, 'You have problems with internet connection'


def test_numpy():
    import numpy as np
    test_array = np.array([1, 2, 3, 4])
    assert test_array.max() == 4, 'numpy.max broken'


def test_pandas():
    import pandas as pd
    from numpy import dtype
    s1 = pd.Series([1, 2, 3, 4, 5, 6])
    assert s1.dtype == dtype('int64'), 'pandas invalid type detection'
    s2 = pd.Series([1.2, 3.4, 5.6])
    assert s2.dtype == dtype('float64'), 'pandas invalid type detection'


def test_beautiful_soup():
    import bs4
    soup = bs4.BeautifulSoup('<html><body><p>Test</p></body></html>', 'html.parser')
    assert len(soup.find_all('p')) == 1, 'bs4 bad data search'
    assert soup.p.string == 'Test', 'bs4 bad data extraction'


def test_scipy():
    import numpy as np
    import scipy.linalg as linalg
    series = np.array([[1, 2], [3, 4]])
    matrix = np.dot(series, series.T)
    assert linalg.svd(matrix), 'SciPy linalg.svd failed'


def test_sklearn():
    from sklearn import datasets, svm
    digits = datasets.load_digits()
    clf = svm.SVC(gamma=0.001, C=100.)
    clf.fit(digits.data[:-1], digits.target[:-1])
    predict = clf.predict(digits.data[-1:])
    assert predict[0] == 8, 'sklearn missclassified'


def test_matplotlib():
    import matplotlib.pyplot as plt
    from sklearn import datasets
    digits = datasets.load_digits()
    plt.figure(1, figsize=(4, 4))
    plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()


def test_seaborn():
    import seaborn as sns
    sns.set(style="ticks")

    # Load the example dataset for Anscombe's quartet
    df = sns.load_dataset("anscombe")

    # Show the results of a linear regression within each dataset
    sns.lmplot(x="x", y="y", col="dataset", hue="dataset", data=df,
               col_wrap=2, ci=None, palette="muted", size=4,
               scatter_kws={"s": 50, "alpha": 1})

    sns.plt.show()


def test_all():
    test_methods = [test_requests, test_numpy, test_pandas,
                    test_beautiful_soup, test_scipy, test_sklearn,
                    test_matplotlib, test_seaborn]
    for method in test_methods:
        try:
            method()
            print '{} OK'.format(method.__name__)
        except Exception as e:
            print '{} failed with exception: {}'.format(method.__name__, str(e))


if __name__ == '__main__':
    test_all()
