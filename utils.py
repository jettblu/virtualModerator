import numpy as np
import struct
import random
from tkinter import filedialog
import pyttsx3
import csv
import string


def float_to_byte(sig):
    # float32 -> int16(PCM_16) -> byte
    return float2pcm(sig, dtype='int16').tobytes()


def float2pcm(sig, dtype='int16'):
    """Convert floating point signal with a range from -1 to 1 to PCM.
    Any signal values outside the interval [-1.0, 1.0) are clipped.
    No dithering is used.
    Note that there are different possibilities for scaling floating
    point numbers to PCM numbers, this function implements just one of
    them.  For an overview of alternatives see
    http://blog.bjornroche.com/2009/12/int-float-int-its-jungle-out-there.html
    Parameters
    ----------
    sig : array_like
        Input array, must have floating point type.
    dtype : data type, optional
        Desired (integer) data type.
    Returns
    -------
    numpy.ndarray
        Integer data, scaled and clipped to the range of the given
        *dtype*.
    See Also
    --------
    pcm2float, dtype
    """
    sig = np.asarray(sig)
    if sig.dtype.kind != 'f':
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in 'iu':
        raise TypeError("'dtype' must be an integer type")

    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)


def floatArraytoPCM(toConvert):
    samples = [sample * 32767
               for sample in toConvert]
    return struct.pack("<%dh" % len(samples), *samples)


def getAudioPath():
    fileName = filedialog.askopenfilename(filetypes=(("Audio Files", ".wav .ogg"), ("All files", "*.*")))
    return fileName


def randomColors(numSpeakers):
    distinctColors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
                      '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
                      '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
                      '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',
                      '#000000']
    random.shuffle(distinctColors)
    return distinctColors[0:numSpeakers]


# enables audio notifications
def verbalSuggestions(cues, isMale=False, rate=146):
    if isMale:
        isMale = 0
    else:
        isMale = 1
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('rate', rate)
    engine.setProperty('voice', voices[isMale].id)
    for cue in cues:
        # que cue
        engine.say(cue)
    engine.runAndWait()

