import torch
import time
import numpy as np
from virtualModerator.language import langUtils as utils
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import joblib
from imblearn.over_sampling import SMOTE


conllData = utils.readConll()
beg = time.time()
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
roberta.eval()
print(f'{time.time()-beg} seconds to load Roberta')


def getTrainFeaturesAndLabels(useStored=False, store=True):
    keyErrors = 0
    if useStored:
        trainLabels = np.load('train labels.npy')
        trainFeatures = np.load('train features.npy')
        return trainFeatures, trainLabels
    errors = 0
    start = time.time()
    trainLabels = []
    trainFeatures = []
    with torch.no_grad():
        count = 0
        for chunk in conllData.trainChunks:
            count += 1
            text = chunk.fullText
            # try logic ensures roberta is actually able to extract features
            try:
                doc = roberta.extract_features_aligned_to_words(text)
            except AssertionError:
                errors += 1
                print(f'Total Errors: {errors} \nThis error occurred while extracting: {chunk.fullText}')
                continue
            doc = [tok for tok in doc if str(tok) != '</s>' and str(tok) != '<s>']
            for tok in doc:
                # try logic cases for mismatch in Roberta tokens and data set tokens
                try:
                    label = chunk.posLabels[str(tok)]
                    trainLabels.append(label)
                    trainFeatures.append(np.array(tok.vector))
                except KeyError:
                    print(f'Key mismatch: {tok} {chunk.posLabels}')
                    keyErrors += 1
            print(f'{count}/{len(conllData.trainChunks)}')

    print(f'{time.time() - start} seconds to collect train features and labels')
    print(f'Key errors: {keyErrors} ')
    trainFeatures = np.array(trainFeatures)
    trainLabels = np.array(trainLabels)

    if store:
        np.save('train features.npy', trainFeatures)
        np.save('train labels.npy', trainLabels)
    return trainFeatures, trainLabels


def trainModels(storeModels=True):
    trainFeatures, trainLabels = getTrainFeaturesAndLabels(useStored=True)
    sm = SMOTE(random_state=2)
    trainFeatures, trainLabels = sm.fit_sample(trainFeatures, trainLabels)
    print(trainFeatures, trainLabels)
    clfs = [KNeighborsClassifier(n_neighbors=10), KNeighborsClassifier(n_neighbors=20)]
    for clf in clfs:
        start = time.time()
        clf.fit(trainFeatures, trainLabels)
        print(f'{time.time() - start} seconds to train {clf}')
        if storeModels:
            joblib.dump(clf, f'{clf}.pkl')


def predict(text):
    with torch.no_grad():
        doc = roberta.extract_features_aligned_to_words(text)
    clf = joblib.load('KNeighborsClassifier(n_neighbors=10).pkl')
    posTagsDict = {0: 'B-INTJ', 1: 'B-LST', 2: 'B-PRT', 3: 'I-UCP', 4: 'B-CONJP', 5: 'B-NP', 6: 'I-ADVP', 7: 'I-PP',
                   8: 'I-INTJ', 9: 'B-SBAR', 10: 'B-PP', 11: 'I-SBAR', 12: 'B-VP', 13: 'B-ADJP', 14: 'I-NP',
                   15: 'B-UCP', 16: 'I-PRT', 17: 'O', 18: 'I-CONJP', 19: 'I-ADJP', 20: 'I-VP', 21: 'B-ADVP', 22: 'I-LST'}
    for tok in doc:
        pred = clf.predict([np.array(tok.vector)])[0]
        tag = posTagsDict[pred]
        print(f'{str(tok)}-----{tag}')


# snips = ['My name is Jett.', 'I am a rockstar', 'You blow my mind.', 'Donald Trump has orange hair.']
# for snip in snips:
#     predict(snip)


labelDict = dict()

labels = np.load('train labels.npy')
for label in labels:
    labelDict[label] = labelDict.get(label, 0) + 1

print(labelDict)