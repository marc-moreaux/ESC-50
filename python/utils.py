import pandas as pd
import numpy as np
import scipy.io.wavfile as sci_wav
from random import shuffle
import os


def change_audio_rate(audio_fname, directory, new_audio_rate):
    '''If the desired file doesn't exist, calls ffmpeg to change the sample rate
    of an audio file.
    eg : change_audio_rate('audio.wav', '/tmp/', 16000)

    Parameters
    ----------
    audio_fname : str
        name of the audio file
    directory : str
        Directory where the audio is stored
    new_audio_rate : int
        Desired sample rate
    '''
    import subprocess
    new_directory = os.path.join(directory, str(new_audio_rate))
    wav_path_orig = os.path.join(directory, audio_fname)
    wav_path_dest = os.path.join(new_directory, audio_fname)
    if not os.path.isfile(wav_path_dest):
        if not os.path.isdir(new_directory):
            os.mkdir(new_directory)
        cmd = 'ffmpeg -i {} -ar {} -b:a 16k -ac 1 {}'.format(
            wav_path_orig,
            new_audio_rate,
            wav_path_dest)
        subprocess.call(cmd, shell=True)


class ESC50(object):
    """This class is shipped with a generator yielding audio from ESC10 or
    ESC50. You may specify the folds you want to used

    eg:
    train = ESC50(folds=[1,2,3])
    train.data_gen.next()

    Parameters
    ----------
    folds : list of integers
        The folds you want to load

    only_ESC10 : boolean
        Wether to use ESC10 instead of ESC50
    """
    def __init__(self, folds=[1,2], only_ESC10=False, randomize=True,
                 audio_rate=44100):
        self.audio_rate = audio_rate
        self.csv_path = '../meta/esc50.csv'
        self.wav_dir = '../audio'
        self.df = pd.read_csv(self.csv_path)
        self.df[self.df.fold.isin(folds)]
        if only_ESC10 is True:
            self.df[self.df.fold.isin(folds)]
        self.data_gen = self._data_gen(randomize)

    def _data_gen(self, randomize=True):
        self.stop = False
        while not self.stop:
            idxs = list(range(len(self.df)))
            if randomize is True:
                shuffle(idxs)

            for idx in idxs:
                fname = self.df.filename[idx]
                change_audio_rate(fname, self.wav_dir, self.audio_rate)
                fpath = os.path.join(self.wav_dir, fname)
                wav_freq, wav_data  = sci_wav.read(fpath)
                wav_data = self.pre_process(wav_data)
                yield wav_data, self.df.target[idx]

    def pre_process(self,
                    audio,
                    pad=0,
                    random_crop=False,
                    normalize=False,
                    random_gain=0
                    random_scale=0
                    ):
        """Apply desired pre_processing to the input

        Parameters
        ----------
        amplitude_threshold : float
            Filters out begining and tail of audio signal with amplitude
            below <amplitude_threshold> of audio.max()
        """
        preprocess_funcs = []
        for f in self.preprocess_funcs:
            audio = f(audio)

        return audio


# Test
train = ESC50(folds=[1,2,3], audio_rate=16000)
next(train.data_gen)
