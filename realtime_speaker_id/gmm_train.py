import numpy as np
import librosa
import pyaudio
import math


from sklearn import svm
from sklearn.mixture import GaussianMixture


def extractGMMFeature(rawSound, samplingRate=22050):
    # MFCC configuration:
    N_MFCC = 55
    N_MELS = 15
    WINDOW_LENGTH = int(0.02 * samplingRate)  # To obtain 20 ms window
    HOP_LENGTH = int(0.010 * samplingRate)  # 10 ms shift between consecutive windows

    # Extracting MFCCs:
    mfccs = librosa.feature.mfcc(rawSound, sr=samplingRate, n_mfcc=N_MFCC, n_mels=N_MELS, n_fft=WINDOW_LENGTH,
                                 hop_length=HOP_LENGTH).T
    deltaMfcc = librosa.feature.delta(mfccs, order=1)
    deltaMfcc2nd = librosa.feature.delta(mfccs, order=2)
    result = np.hstack((mfccs, deltaMfcc, deltaMfcc2nd))
    print(result.shape)
    return result


def trainGMM(filename, nMixtures=50):
    rawData = librosa.load(filename)[0]
    features = extractGMMFeature(rawData)
    gmm = GaussianMixture(n_components=nMixtures)
    start = 0
    while start + 200 < features.shape[0]:
        gmm.fit(features[start:start + 200, :])
        start += 200
    return gmm


def score_gmm(gmm, fileName):
    features = extractGMMFeature(fileName)
    return gmm.score(features)


gmms = []

gmms.append(trainGMM('trimmed.wav'))
# gmms.append(trainGMM('trimmed_2.wav'))
gmms.append(trainGMM('jett 1.wav'))
# gmms.append(trainGMM('trimmed_3.wav'))
# gmms.append(trainGMM('trimmed_4.wav'))

CHUNKSIZE = 22050

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=22050, input=True, frames_per_buffer=CHUNKSIZE)


print('Starting Microphone')
while True:
    data = stream.read(CHUNKSIZE)
    data = np.frombuffer(data, dtype=np.float32)
    feature = extractGMMFeature(data)
    max_prob = -math.inf
    max_id = -1
    for i in range(len(gmms)):
        score = gmms[i].score(feature)
        print(i, score)





