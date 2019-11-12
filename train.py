import json
import os
import re

import numpy
import tensorflow.keras

import params

url_re = re.compile(' ?https://t[.]co/[a-zA-Z0-9]*')
def get_text(obj):
    if obj['is_retweet']:
        return ''
    return quote(url_re.sub('', obj['text']))

def quote(str):
    q = ''
    for i in range(len(str)):
        if ord(str[i]) < 32 or ord(str[i]) > 126:
            if q == '':
                q = str[:i]
                pass
            q += '\\u{:04x}'.format(ord(str[i]))
            pass
        elif q != '':
            q += str[i]
            pass
        pass
    if q == '':
        return str
    return q

tws = []
for file in ['condensed_2018.json', 'condensed_2017.json', 'condensed_2016.json']:
    with open(file, 'r') as f:
        tws += json.load(f, object_hook=get_text)
        pass
    pass

size = 0
for t in tws:
    if t != '':
        size += 1 + len(t)
        pass
    pass

def train(filename, maxlen, step):
    global size, tws
    ntrain = size // step + 1
    window = [95 for i in range(maxlen+2)]
    x = numpy.zeros((ntrain, maxlen, 96), dtype=numpy.bool)
    y = numpy.zeros((ntrain, 2, 96), dtype=numpy.bool)
    iwindow = 0
    def push_window(c):
        nonlocal maxlen, step
        nonlocal window, x, y, iwindow
        window = window[1:]
        window.append(c)
        iwindow += 1
        if iwindow % step == 0:
            for i in range(maxlen):
                x[iwindow // step, i, window[i+1]] = 1
                pass
            y[iwindow // step, 0, window[0]] = 1
            y[iwindow // step, 1, window[maxlen+1]] = 1
            pass
        pass

    for itw in range(len(tws)):
        tw = tws[len(tws)-1-itw]
        if tw == '':
            continue
        push_window(95)
        if iwindow // step >= ntrain:
            break
        for c in tw:
            push_window(ord(c)-32)
            if iwindow // step >= ntrain:
                break
            pass
        if iwindow // step >= ntrain:
            break
        pass

    try:
        os.stat(filename)
        model = tensorflow.keras.models.load_model(filename)
    except:
        model = tensorflow.keras.models.Sequential()
        model.add(tensorflow.keras.layers.LSTM(128, input_shape=(maxlen, 96)))
        model.add(tensorflow.keras.layers.Dense(2*96, activation='softmax'))
        model.add(tensorflow.keras.layers.Reshape((2,96)))
        model.compile(loss='categorical_crossentropy', optimizer=tensorflow.keras.optimizers.RMSprop(learning_rate=0.01))
        pass
    model.fit(x, y, epochs=10, batch_size=4096)
    model.save(filename)
    del model
    pass

train(params.filename_short, params.maxlen_short, params.step_short)
train(params.filename_long, params.maxlen_long, params.step_long)
