class Logger(object):
    def __init__(self, filename):
        self.file = filename
        with open(self.file, 'w') as f:
            f.write("")
        
    def log(self, content):
        print(content)
        with open(self.file, 'a') as f:
            f.write(content + '\n')