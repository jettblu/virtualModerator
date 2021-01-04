import torch
import time
import numpy as np
from virtualModerator.language import langUtils as utils
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
import joblib


sentimentData = utils.readSentimentSets()
beg = time.time()
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
roberta.eval()
print(f'{time.time()-beg} seconds to load Roberta')


def getFeatures(sentences):
    with torch.no_grad():
        allTokens = []
        for sent in sentences:
            tokens = roberta.encode(sent)
            allTokens.append(list(tokens))
        # max number of tokens Roberta can take
        maxLength = 500
        padded = [torch.tensor(tokenSet + [0] * (maxLength-len(tokenSet))) for tokenSet in allTokens]
        features = []
        progressCount = 0
        layersHolder = []
        for tokenSet in padded:
            progressCount += 1
            print(f'{progressCount}/{len(allTokens)}')
            try:
                allLayers = roberta.extract_features(tokenSet, return_all_hiddens=True)
                layersHolder.append(allLayers)
            except ValueError:
                print(f'Error processing {tokenSet}')
            feature = []
            for layer in allLayers:
                feature.append(layer[0][0][0])
            features.append(np.array(feature))
    return features, layersHolder


def getFeaturesAndLabels(sentimentHolder, useStored=False, store=True):
    start = time.time()
    if useStored:
        labels = np.load('sentiment labels.npy')
        features = np.load('sentiment features.npy')
        return features, labels
    labels = []
    for _ in sentimentHolder.positive:
        labels.append(1)
    features, layersHolder = getFeatures(sentimentHolder.positive)
    for _ in sentimentHolder.negative:
        labels.append(0)
    features2, layersHolder2 = getFeatures(sentimentHolder.negative)
    features.extend(features2)

    print(f'{time.time() - start} seconds to collect train features and labels')

    features = np.array(features)
    labels = np.array(labels)
    if store:
        np.save('sentiment features.npy', features)
        np.save('sentiment labels.npy', labels)
        np.save('positive sentiment layers.npy', layersHolder)
        np.save('negative sentiment layers.npy', layersHolder2)

    return features, labels


def getAccuracy(classifier, data, labels, storeModel=True):
    start = time.time()
    testScores = []
    # set up container for class level results of each classification trial
    cv = KFold(n_splits=10, random_state=65, shuffle=True)
    # make predictions
    for train_index, test_index in cv.split(data):
        dataTrain, dataTest, labelsTrain, labelsTest = data[train_index], data[test_index], labels[train_index], labels[test_index]
        classifier.fit(dataTrain, labelsTrain)
        # create confusion matrix
        testScores.append(classifier.score(dataTest, labelsTest))
    if storeModel:
        joblib.dump(classifier, f'sentiment {classifier}.pkl')
    print(f'{time.time() - start} seconds to train {classifier}')
    return np.mean(testScores)


def trainModels(useStored=True, store=True):
    features, labels = getFeaturesAndLabels(sentimentHolder=sentimentData, useStored=useStored)
    clfs = [KNeighborsClassifier(n_neighbors=10), KNeighborsClassifier(n_neighbors=20), SVC(kernel='rbf')]
    accuracies = dict()
    for clf in clfs:
        accuracy = getAccuracy(classifier=clf, data=features, labels=labels)
        accuracies[str(clf)] = accuracy
    return accuracies


print(trainModels(useStored=False))

