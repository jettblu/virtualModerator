import random
import csv

class Conll:
    posTagsDict = {'B-INTJ': 0, 'B-LST': 1, 'B-PRT': 2, 'I-UCP': 3, 'B-CONJP': 4, 'B-NP': 5, 'I-ADVP': 6, 'I-PP': 7,
                   'I-INTJ': 8, 'B-SBAR': 9, 'B-PP': 10, 'I-SBAR': 11, 'B-VP': 12, 'B-ADJP': 13, 'I-NP': 14,
                   'B-UCP': 15, 'I-PRT': 16, 'O': 17, 'I-CONJP': 18, 'I-ADJP': 19,
                   'I-VP': 20, 'B-ADVP': 21, 'I-LST': 22}
    testChunks = []
    trainChunks = []


class Chunk:
    def __init__(self, chunk, wsjLabels, posLabels):
        self.fullText = chunk
        self.wsjLabels = wsjLabels
        self.posLabels = posLabels


def readTestConll():
    testChunk = ''
    testWsjLabels = []
    testPosLabels = dict()
    with open('language/language data sets/conll 2000 test.txt', newline='') as csvfile:
        dataReader = csv.reader(csvfile, delimiter=' ')
        for row in dataReader:
            if not row:
                # store compiled chunk...reset holders
                Conll.testChunks.append(Chunk(chunk=testChunk, wsjLabels=testWsjLabels, posLabels=testPosLabels))
                testChunk = ''
                testWsjLabels = []
                testPosLabels = dict()
            else:
                # update current chunk
                word = row[0]
                wsjLabel = row[1]
                posLabel = row[2]
                if testChunk == '':
                    testChunk = word
                else:
                    testChunk += f' {word}'
                testWsjLabels.append(wsjLabel)
                testPosLabels[word] = Conll.posTagsDict[posLabel]


def readTrainConll():
    trainChunk = ''
    trainWsjLabels = []
    trainPosLabels = dict()
    with open('language/language data sets/conll 2000 train.txt', newline='') as csvfile:
        dataReader = csv.reader(csvfile, delimiter=' ')
        for row in dataReader:
            if not row:
                # store compiled chunk...reset holders
                Conll.trainChunks.append(Chunk(chunk=trainChunk, wsjLabels=trainWsjLabels, posLabels=trainPosLabels))
                trainChunk = ''
                trainWsjLabels = []
                trainPosLabels = dict()
            else:
                # update current chunk
                word = row[0]
                wsjLabel = row[1]
                posLabel = row[2]
                if trainChunk == '':
                    trainChunk = word
                else:
                    trainChunk += f' {word}'
                trainWsjLabels.append(wsjLabel)
                trainPosLabels[word] = Conll.posTagsDict[posLabel]


class SentimentHolder:
    def __init__(self):
        self.positive = []
        self.negative = []

    def reorganize(self):
        minLen = min(len(self.positive), len(self.negative))
        self.positive = self.positive[:minLen]
        self.negative = self.negative[:minLen]
        random.shuffle(self.positive)
        random.shuffle(self.negative)


def readSentimentSets():
    sentimentHolder = SentimentHolder()

    def process(dataReader):
        for row in dataReader:
            sentence = row[0].lower()
            sentiment = row[1]
            if sentiment == '1':
                sentimentHolder.positive.append(sentence)
            else:
                sentimentHolder.negative.append(sentence)
    with open('language data sets/sentiment/labeledSentences/amazon_cells_labelled.txt', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        process(reader)
    with open('language data sets/sentiment/labeledSentences/imdb_labelled.txt', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        process(reader)
    with open('language data sets/sentiment/labeledSentences/yelp_labelled.txt', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        process(reader)

    sentimentHolder.reorganize()
    return sentimentHolder


def readConll():
    readTrainConll()
    readTestConll()
    return Conll
