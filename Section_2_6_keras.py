""" Часть I. Глава 2. Раздел 2.6. Линейная регрессия"""


import numpy as np
from keras.models import Sequential
from keras.layers import Input, Dense, Activation

# модель будет создаваться последовательно слой за слоем
logr = Sequential()
# добавляем один плотный слой, входы которого будут размерности два,
# а на выходе будет логистический сигмоид от входов
logr.add(Dense(1, input_dim=2, activation= 'sigmoid' ))
# модель компилируется с заданной целевой функцией и метрикой точности
logr.compile(loss= 'binary_crossentropy' , optimizer= 'sgd' , metrics=[ 'accuracy' ])

def sampler(n, x, y):
    return np.random.normal(size=[n, 2]) + [x, y]

# генерируем два двумерных нормальных распределения, одно с центром в
# точке (−1; −1), другое в точке (1; 1), с дисперсией 1 по обоим компонентам
def sample_data(n=1000, p0=(-1., -1.), p1=(1., 1.)):
    zeros, ones = np.zeros((n, 1)), np.ones((n, 1))
    labels = np.vstack([zeros, ones])
    z_sample = sampler(n, x=p0[0], y=p0[1])
    o_sample = sampler(n, x=p1[0], y=p1[1])
    return np.vstack([z_sample, o_sample]), labels

X_train, Y_train = sample_data()
X_test, Y_test = sample_data(100)

#задаем тренировочное множество, число эпох и размер мини-батча
logr.fit(X_train, Y_train, batch_size=16, nb_epoch=100,
    verbose=1, validation_data=(X_test, Y_test))
