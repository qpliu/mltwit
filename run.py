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

def sample(preds, temperature=0.35):
    preds = numpy.asarray(preds).astype('float64')
    preds = numpy.log(preds) / temperature
    exp_preds = numpy.exp(preds)
    preds = exp_preds / numpy.sum(exp_preds)
    probas = numpy.random.multinomial(1, preds, 1)
    return numpy.argmax(probas)

models = list(map(lambda p: tensorflow.keras.models.load_model(p.filename), params.params))
xs = list(map(lambda p: numpy.zeros((1,p.maxlen,96)), params.params))

for text in sys.argv[1:]:
    have_start = False
    have_end = False
    start_dots = 0
    end_dots = 0
    for _ in range(200):
        x = xs[0]
        model = models[0]
        maxlen = params.params[0].maxlen
        delay_head = False
        delay_tail = False
        if len(text) < params.params[0].runlen[1]:
            if text[0].isupper():
                delay_head = True
            else:
                delay_tail = True
                pass
            pass
        for i in range(len(params.params)):
            if len(text) >= params.params[i].runlen[0] and (params.params[i].runlen[1] == None or len(text) < params.params[i].runlen[1]):
                x = xs[i]
                model = models[i]
                maxlen = params.params[i].maxlen
                break
            pass
        if not have_start and (not delay_head or have_end):
            temperature = 0.2
            if text[0] == ' ':
                temperature = 0.5
                pass
            x.fill(0)
            for i in range(min(len(text), maxlen)):
                c = min(95,max(0,ord(text[i])-32))
                x[0,i,c] = 1
                pass
            y = model.predict(x, verbose=0)
            c = sample(y[0,0,:], temperature=temperature)
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
        if not have_end and (not delay_tail or have_start):
            temperature = 0.2
            if text[-1] == ' ':
                temperature = 0.5
                pass
            x.fill(0)
            imax = min(len(text), maxlen)
            for i in range(imax):
                c = min(95,max(0,ord(text[len(text)-imax+i])-32))
                x[0,maxlen-imax+i,c] = 1
                pass
            y = model.predict(x, verbose=0)
            c = sample(y[0,1,:], temperature=temperature)
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
