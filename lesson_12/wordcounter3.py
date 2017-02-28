from mrjob.job import MRJob
from collections import defaultdict

class mrWordCount(MRJob):
    def __init__(self, *args, **kwargs):
        super(mrWordCount, self).__init__(*args, **kwargs)
        self.localWordCount = defaultdict(int)
    
    def mapper(self, key, line):
        if False:
            yield
        for word in line.split(' '):
            self.localWordCount[word.lower()] += 1
            
    def mapper_final(self):
        for (word, count) in self.localWordCount.iteritems():
            yield word, count

    def reducer(self, word, occurences):
        yield word, sum(occurences)


if __name__ == '__main__':
    mrWordCount.run()