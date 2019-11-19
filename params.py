class Params:
    def __init__(self, maxlen=10, step=3, filename=None, epochs=1, runlen=(0,20)):
        if filename == None:
            filename = 'model-{:d}.h5'.format(maxlen)
            pass
        self.maxlen = maxlen
        self.step = step
        self.filename = filename
        self.epochs = epochs
        self.runlen = runlen
        pass
    pass

params = [
    Params(maxlen=10, runlen=(0,20), epochs=5),
    Params(maxlen=40, runlen=(20,50), epochs=5),
    Params(maxlen=80, runlen=(50,None), epochs=5),
    ]
