import tkinter as tk
import speakerIdentification
import pickle
from tkinter import simpledialog
from tkinter import scrolledtext

from moderate import *
import utils
from moderatorWindow import ModeratorWindow


def wait(window, message):
    win = tk.Toplevel(window)
    win.transient()
    win.title('Wait')
    tk.Label(win, text=message).pack()
    window.update()
    return win


def createSpeakerCollection(window):
    spkr_cnt = simpledialog.askinteger('Input', 'How many speakers are there?',
                                       parent=window,
                                       minvalue=1, maxvalue=100)

    gmms = []
    names = []
    for i in range(spkr_cnt):
        name = simpledialog.askstring('Input',
                                      'What\'s the name of Speaker #{}?'.format(i + 1), parent=window)
        filename = tk.filedialog.askopenfilename(title='Select training audio for {}'.format(name),
                                                 filetypes=(("Audio Files", ".wav .ogg"), ("All files", "*.*")))

        overlay = wait(window, 'Training GMM mixtures for {}'.format(name))
        names.append(name)
        gmms.append(speakerIdentification.trainGMM(filename))

        overlay.destroy()

    # dump speaker collection
    filename = tk.filedialog.asksaveasfilename(title='Save trained speaker collection',
                                               filetypes=(("Speaker collection", ".pkl"), ("All files", "*.*")),
                                               defaultextension='.pkl')

    obj = {'names': names, 'gmms': gmms}
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)

    # create speaker objects
    colors = utils.randomColors(len(names))
    for i, name in enumerate(names):
        Speaker(speakerId=name, color=colors[i])

    return gmms


def loadSpeakerCollection(filename):
    obj = None
    with open(filename, 'rb') as file:
        obj = pickle.load(file)

    names = obj['names']
    gmms = obj['gmms']

    # create speaker objects
    colors = utils.randomColors(len(names))
    for i, name in enumerate(names):
        Speaker(speakerId=name, color=colors[i])

    return gmms


def initializeSpeakerCollection():
    gmms = []

    window = tk.Tk()
    window.title('Speaker Setup')
    window.geometry('300x100')
    window.resizable(0, 0)

    top_frame = tk.Frame(window)
    top_frame.pack()

    data_src = tk.IntVar()
    data_src.set(2)

    label1 = tk.Label(top_frame, text='Please select speaker data source')
    label1.pack(side=tk.LEFT)

    opt_frame = tk.Frame(window)
    opt_frame.pack()

    opt1 = tk.Radiobutton(opt_frame, text='Create a new speaker collection file', variable=data_src, value=1)
    opt1.pack()
    opt2 = tk.Radiobutton(opt_frame, text='Load an exisiting speaker collection', variable=data_src, value=2)
    opt2.pack()

    def okPressed():
        nonlocal gmms

        selected_option = data_src.get()
        if selected_option == 1:
            gmms = createSpeakerCollection(window)
            window.destroy()
        else:
            # load existing file
            filename = tk.filedialog.askopenfilename(filetypes=(("Speaker collection", ".pkl"), ("All files", "*.*")))
            gmms = loadSpeakerCollection(filename)
            window.destroy()

    btn_frame = tk.Frame(window)
    btn_frame.pack(side=tk.BOTTOM)
    button = tk.Button(btn_frame, text='OK', command=okPressed)
    button.pack()

    window.mainloop()
    return gmms


if __name__ == '__main__':
    utils.verbalSuggestions(['Virtual Moderator Initiated!'])
    gmms = initializeSpeakerCollection()
    print('Virtual Moderator Initiated!')
    # See http://g.co/cloud/speech/docs/languages
    # for a list of supported languages.
    language_code = "en-US"  # a BCP-47 language tag
    client = speech.SpeechClient()

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code,
        enable_word_time_offsets=True,
        enable_automatic_punctuation=True,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )


    with MicrophoneStream(RATE) as stream:
        stream.gmms = gmms

        fig = plt.figure()  # the figure will be reused later
        fig.suptitle(f'Moderator Beta', fontsize=16)
        axe = fig.add_subplot(111)

        t2 = threading.Thread(target=analyze, args=(stream,))
        t3 = threading.Thread(target=visualize, args=(fig, axe, stream))
        t4 = threading.Thread(target=ModeratorWindow.task, args=())

        t2.start()
        t3.start()
        t4.start()

        audio_generator = stream.generator()
        requests = (
            speech.StreamingRecognizeRequest(audio_content=content) for content in audio_generator
        )

        responses = client.streaming_recognize(streaming_config, requests)

        t1 = threading.Thread(target=transcribe, args=(responses, stream))

        # start threads
        t1.start()

        plt.show()

        # wait until threads finish their job
        t1.join()
        t2.join()
        t3.join()
        t4.join()

