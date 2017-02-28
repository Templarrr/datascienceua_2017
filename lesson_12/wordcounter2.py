from mrjob.job import MRJob


class mrWordCount(MRJob):
    def mapper(self, key, line):
        for word in line.split(' '):
            yield word.lower(), 1
            
    def combiner(self, word, occurences):
        yield word, sum(occurences)

    def reducer(self, word, occurences):
        yield word, sum(occurences)


if __name__ == '__main__':
    mrWordCount.run()