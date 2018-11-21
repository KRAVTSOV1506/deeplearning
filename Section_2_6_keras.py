import numpy as np
from keras.models import Sequential
from keras.layers import Input, Dense, Activation

#модель будет создаваться последовательно слой за слоем
logr = Sequential()
#добавляем один плотный слой, входы которого будут размерности два,
# а на выходе будет логистический сигмоид от входов
logr.add(Dense(1, input_dim=2, activation= ' sigmoid ' ))
#модель собственно компилируется с заданной целевой функцией и метрикой точности
logr.compile(loss= ' binary_crossentropy ' , optimizer= ' sgd ' , metrics=[ ' accuracy ' ])