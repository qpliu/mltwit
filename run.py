import re
import sys

import numpy
import tensorflow.keras

import params

quoted_re = re.compile(r'\\u([a-f0-9][a-f0-9][a-f0-9][a-f0-9])')
def repl_quoted(m):
    return chr(int(m.group(1),16))
def unquote(text):
    return quoted_re.sub(repl_quoted,text)

model_long = tensorflow.keras.models.load_model(params.filename_long)
model_short = tensorflow.keras.models.load_model(params.filename_short)

def sample(preds, temperature=0.35):
    preds = numpy.asarray(preds).astype('float64')
    preds = numpy.log(preds) / temperature
    exp_preds = numpy.exp(preds)
    preds = exp_preds / numpy.sum(exp_preds)
    probas = numpy.random.multinomial(1, preds, 1)
    return numpy.argmax(probas)

x_long = numpy.zeros((1, params.maxlen_long, 96))
x_short = numpy.zeros((1, params.maxlen_short, 96))

maxlen_transition = max(params.maxlen_long - params.maxlen_short, (params.maxlen_long + params.maxlen_short) // 2)

for text in sys.argv[1:]:
    have_start = False
    have_end = False
    start_dots = 0
    end_dots = 0
    for _ in range(200):
        if not have_start:
            if len(text) < maxlen_transition:
                x_short.fill(0)
                for i in range(min(len(text), params.maxlen_short)):
                    c = min(95,max(0,ord(text[i])-32))
                    x_short[0,i,c] = 1
                    pass
                y = model_short.predict(x_short, verbose=0)
                c = sample(y[0,0,:])
                pass
            else:
                x_long.fill(0)
                for i in range(min(len(text), params.maxlen_long)):
                    c = min(95,max(0,ord(text[i])-32))
                    x_long[0,i,c] = 1
                    pass
                y = model_long.predict(x_long, verbose=0)
                c = sample(y[0,0,:])
                pass
            if c == 95:
                if start_dots < 2:
                    have_start = True
                else:
                    text = '\n' + text
                    pass
            else:
                if c == ord('.') - 32:
                    start_dots += 1
                else:
                    start_dots = 0
                    pass
                text = chr(32+c) + text
                pass
            pass
        if not have_end:
            if len(text) < maxlen_transition:
                x_short.fill(0)
                imax = min(len(text), params.maxlen_short)
                for i in range(imax):
                    c = min(95,max(0,ord(text[len(text)-imax+i])-32))
                    x_short[0,params.maxlen_short-imax+i,c] = 1
                    pass
                y = model_short.predict(x_short, verbose=0)
                c = sample(y[0,1,:])
                pass
            else:
                x_long.fill(0)
                imax = min(len(text), params.maxlen_long)
                for i in range(imax):
                    c = min(95,max(0,ord(text[len(text)-imax+i])-32))
                    x_long[0,params.maxlen_long-imax+i,c] = 1
                    pass
                y = model_long.predict(x_long, verbose=0)
                c = sample(y[0,1,:])
                pass
            if c == 95:
                if end_dots < 2:
                    have_end = True
                else:
                    text += '\n'
                    pass
            else:
                if c == ord('.') - 32:
                    end_dots += 1
                else:
                    end_dots = 0
                    pass
                text += chr(32+c)
                pass
            pass
        if have_start and have_end:
            break
        pass
    print(unquote(text))
    pass
