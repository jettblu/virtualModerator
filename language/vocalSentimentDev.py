import librosa
import os
import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
import joblib


def extractAudioFeatures(rawSound, samplingRate=22050):
    # MFCC configuration:
    N_MFCC = 55
    N_MELS = 15
    WINDOW_LENGTH = int(1 * samplingRate)  # To obtain 25 ms window
    HOP_LENGTH = int(0.010 * samplingRate)  # 10 ms shift between consecutive windows

    # Extracting MFCCs:
    mfccs = librosa.feature.mfcc(rawSound, sr=samplingRate, n_mfcc=N_MFCC, n_mels=N_MELS, n_fft=WINDOW_LENGTH,
                                 hop_length=HOP_LENGTH).T
    deltaMfcc = librosa.feature.delta(mfccs, order=1)
    deltaMfcc2nd = librosa.feature.delta(mfccs, order=2)
    result = np.hstack((mfccs, deltaMfcc, deltaMfcc2nd))
    return result


def organizePathNames():
    # FUNCTION TO EXTRACT EMOTION NUMBER, ACTOR AND GENDER LABEL
    emotions = {1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}
    dataSetEmotions = {'neutral': [], 'calm': [], 'happy': [], 'sad': [], 'angry': [], 'fear': [], 'disgust': [], 'surprise': []}
    basePath = 'language data sets/audioSentiment'
    actorFolders = os.listdir(basePath)
    for folderName in actorFolders:
        folderBase = basePath+'/'+folderName
        folder = os.listdir(folderBase)  # iterate over Actor folders
        for fileName in folder:  # go through files in Actor folder
            print(fileName)
            try:
                parts = fileName.split('.')[0].split('-')
                emotionLabel = emotions[int(parts[2])]
                dataSetEmotions[emotionLabel].append(folderBase + '/' + fileName)
            except IndexError:
                print(f'{fileName} not valid')
    return dataSetEmotions


def getFeaturesAndLabels(filesDict, store=True, useStored=False):
    if useStored:
        print('poop')
        labels = np.load('audio sentiment labels.npy')
        features = np.load('audio sentiment features.npy')
        return features, labels
    features = []
    labels = []
    for i, emotion in enumerate(filesDict):
        files = filesDict[emotion]
        for file in files:
            fv = extractAudioFeatures(librosa.load(file)[0])
            for segment in fv:
                features.append(segment)
                labels.append(i)
    if store:
        np.save('audio sentiment features.npy', features)
        np.save('audio sentiment labels.npy', labels)
    print('here')
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
        joblib.dump(classifier, f'audio sentiment {classifier}.pkl')
    print(f'{time.time() - start} seconds to train {classifier}')
    return np.mean(testScores)


def trainModels(useStored=False, store=True):
    features, labels = getFeaturesAndLabels(organizePathNames(), store=store, useStored=useStored)
    clfs = [KNeighborsClassifier(n_neighbors=10), KNeighborsClassifier(n_neighbors=20), SVC(kernel='rbf')]
    accuracies = dict()
    for clf in clfs:
        accuracy = getAccuracy(classifier=clf, data=features, labels=labels)
        accuracies[str(clf)] = accuracy
    return accuracies


def predict(rawData):
    emotions = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'}
    audioFeatures = extractAudioFeatures(rawSound=rawData)
    clf = joblib.load('language/audio sentiment KNeighborsClassifier(n_neighbors=10).pkl')
    # add 1 to account for 0 base in enumeration and 1 base in emotions dict
    predictions = clf.predict(audioFeatures)
    votingCount = {}
    # record votes
    for label in predictions:
        votingCount[label] = votingCount.get(label, 0) + 1
    # tally votes
    prediction = emotions[max(votingCount, key=votingCount.get)+1]
    return prediction
