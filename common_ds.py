# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])


def cv_optimize(clf, parameters, X, y, n_jobs=1, n_folds=5, score_func=None):
    """ Оптимизируем гиперпараметры классификатора с помощью Grid Search.
    Перебираем все сочетания предложенных параметров и смотрим какое из них даст наилучший
    результат (в среднем после кроссвалидации).

    Parameters
    ----------
    clf : sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin
        Объект-классификатор. Предполагается имплементация score и fit методов.
    parameters : dict
        Перебираемые параметры в виде {параметр1:[значение1_1, значение1_2],параметр2:[...]}
    X : numpy.ndarray
        Список спиской, данные признаков классификатора. Каждый элемент списка представляет
        запись исходного датасета.
    y : numpy.ndarray
        Данные целевой колонки
    n_jobs : int
        Количество одновременно запускаемых параллельно задач.
    n_folds : int
        Параметр кроссвалидации
    score_func : string or callable, optional
        Функция оценки результата, принимающая классификатор,
        X и Y (предикторы и целевую колонку).
        Стандартные функции описаны в sklearn Model Evaluation docs.
        При отсутствии используется clf.score

    Returns
    -------
    sklearn.base.ClassifierMixin
        Классификатор с оптимальными (из предоставленных) гиперпараметрами.
    """
    if score_func:
        gs = GridSearchCV(clf, param_grid=parameters, cv=n_folds, n_jobs=n_jobs, scoring=score_func)
    else:
        gs = GridSearchCV(clf, param_grid=parameters, n_jobs=n_jobs, cv=n_folds)
    gs.fit(X, y)
    print "BEST", gs.best_params_, gs.best_score_
    best = gs.best_estimator_
    return best


def do_classify(clf, parameters, indf, featurenames, targetname, target1val, mask=None,
                reuse_split=None, score_func=None, n_folds=5, n_jobs=1):
    """ Провести бинарную классификацию данных с оптимальными параметрами и вывести оценку точности
    классификатора на тренировочных и тестовых данных и confusion matrix.

    Parameters
    ----------
    clf : sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin
        Объект-классификатор.
    parameters : dict
        Перебираемые параметры в виде {параметр1:[значение1_1, значение1_2],параметр2:[...]}
    indf : pandas.DataFrame
        Данные для обучения
    featurenames : list of str
        Заголовки колонок датасета, которые мы используем для обучения
    targetname : str
        Заголовок целевой колонки
    target1val : str
        Значение в целевой колонке которое будет считаться классом 1 (остальные - 0)
    mask : list of bool, optional
        Маска для отделения тестовых данных, от данных обучения.
        True для строк которые будут использованы для обучения, False для оценки.
        При использовании reuse_split игнорируется
    reuse_split : dict, optional
        Использовать предварительно разделенные данные, ключи Xtrain, Xtest, ytrain, ytest.
    score_func : callable, optional
        Функция оценки результата, принимающая векторы Y и Y' (цель обучения и предсказания).
        При отсутствии используется clf.score
    n_jobs : int
        Количество одновременно запускаемых параллельно задач.
    n_folds : int
        Параметр кроссвалидации

    Returns
    -------
    clf : sklearn.base.ClassifierMixin
        Обученый классификатор.
    Xtrain, ytrain, Xtest, ytest : numpy.ndarray
        Данные использованного деления.
    """
    subdf = indf[featurenames]
    X = subdf.values
    y = (indf[targetname].values == target1val) * 1
    if mask != None:
        print "using mask"
        Xtrain, Xtest, ytrain, ytest = X[mask], X[~mask], y[mask], y[~mask]
    if reuse_split != None:
        print "using reuse split"
        Xtrain, Xtest, ytrain, ytest = reuse_split['Xtrain'], reuse_split['Xtest'], reuse_split[
            'ytrain'], reuse_split['ytest']
    if parameters:
        clf = cv_optimize(clf, parameters, Xtrain, ytrain, n_jobs=n_jobs, n_folds=n_folds,
                          score_func=score_func)
    clf = clf.fit(Xtrain, ytrain)
    training_accuracy = clf.score(Xtrain, ytrain)
    test_accuracy = clf.score(Xtest, ytest)
    print "############# based on standard predict ################"
    print "Accuracy on training data: %0.2f" % (training_accuracy)
    print "Accuracy on test data:     %0.2f" % (test_accuracy)
    print confusion_matrix(ytest, clf.predict(Xtest))
    print "########################################################"
    return clf, Xtrain, ytrain, Xtest, ytest


def points_plot(ax, Xtr, Xte, ytr, yte, clf, mesh=True, alpha=0.3, psize=10):
    """ Вывести диаграмму рассеяния по двум первым переменным датасета
    с визуализацией классификатора.
    Точки тренировочных данных обозначены кружками, тестовых - квадратами.

    Parameters
    ----------
    ax : matplotlib.axis.Axes
        Текущий график.
    Xtr : numpy.ndarray
        Тренировочные данные признаков.
    Xte : numpy.ndarray
        Тестовые данные признаков.
    ytr : numpy.ndarray
        Тренировочные данные целевой колонки.
    yte : numpy.ndarray
        Тестовые данные целевой колонки.
    clf : sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin
        Визуализируемый классификатор.
    mesh : bool
        Выводить ли сетку на фоне?
    alpha : float
        Мера непрозрачности точек, от 0 до 1
    psize : int
        Размер точек данных

    Returns
    -------
    ax : matplotlib.axis.Axes
        Текущий график.
    xx, yy : numpy.ndarray
        Значения координатной сетки.
    """
    X = np.concatenate((Xtr, Xte))
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    if mesh:
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light, alpha=alpha, axes=ax)
    ax.scatter(Xtr[:, 0], Xtr[:, 1], c=ytr - 1, cmap=cmap_bold, s=psize, alpha=alpha, edgecolor="k")
    ax.scatter(Xte[:, 0], Xte[:, 1], c=yte - 1, cmap=cmap_bold, s=psize + 10, alpha=alpha, marker="s")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    return ax, xx, yy


def points_plot_prob(ax, Xtr, Xte, ytr, yte, clf,
                     ccolor=cm, psize=10, alpha=0.1, prob=True):
    """

    Parameters
    ----------
    ax : matplotlib.axis.Axes
        Текущий график.
    Xtr : numpy.ndarray
        Тренировочные данные признаков.
    Xte : numpy.ndarray
        Тестовые данные признаков.
    ytr : numpy.ndarray
        Тренировочные данные целевой колонки.
    yte : numpy.ndarray
        Тестовые данные целевой колонки.
    clf : sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin
        Визуализируемый классификатор.
    ccolor : colormap
        Цветовая схема для линий контуров
    alpha : float
        Мера непрозрачности точек, от 0 до 1
    psize : int
        Размер точек данных
    prob : bool
        True для классификаторов возвращающих вероятность, False - для дискриминаторов.

    Returns
    -------
    ax : matplotlib.axis.Axes
        Текущий график.
    """
    ax, xx, yy = points_plot(ax, Xtr, Xte, ytr, yte, clf, mesh=False, psize=psize, alpha=alpha)
    if prob:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    else:
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=ccolor, alpha=.2, axes=ax)
    cs2 = plt.contour(xx, yy, Z, cmap=ccolor, alpha=.6, axes=ax)
    plt.clabel(cs2, fmt='%2.1f', colors='k', fontsize=14, axes=ax)
    return ax
