import pandas as pd
import numpy as np
import bc_utils as U
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
    def __init__(self,
                 folds=[1,2],
                 only_ESC10=False,
                 randomize=True,
                 audio_rate=44100):
        self.audio_rate = audio_rate
        self.csv_path = '../meta/esc50.csv'
        self.wav_dir = '../audio'
        self.df = pd.read_csv(self.csv_path)
        self.df[self.df.fold.isin(folds)]
        if only_ESC10 is True:
            self.df[self.df['esc10']] 
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

    def _preprocess_setup(self,
                          strongAugment=False,
                          pad=0,
                          inputLength=0,
                          normalize=False):
        """Apply desired pre_processing to the input

        Parameters
        ----------
        strongAugment: Bool 
           random scale and put gain in audio input 
        pad:
            add padding before and after audio signal
        inputLength: float
            time in seconds of the audio input
        normalize: float
            value used to normalize input

        """
        funcs = []
        if opt.strongAugment:
            funcs.append(U.random_scale(1.25))

        if opt.pad > 0:
            funcs.append(U.padding(opt.pad))
        
        if opt.inputLength > 0:
            funcs.append(U.random_crop(opt.inputLength))

        if opt.normalize is True:
              U.normalize(32768.0),

    def preprocess(self, audio):
        """Apply desired pre_processing to the input

        Parameters
        ----------
        audio: array 
            audio signal to be preprocess
        """

        for f in self.preprocess_funcs:
            audio = f(audio)

        return audio

    def __len__(self):
        return len(self.df)

    def get_example(self):
        if self.mix:  # Training phase of BC learning
            # Select two training examples
            while True:
                sound1, label1 = self.base[random.randint(0, len(self.base) - 1)]
                sound2, label2 = self.base[random.randint(0, len(self.base) - 1)]
                if label1 != label2:
                    break
            sound1 = self.preprocess(sound1)
            sound2 = self.preprocess(sound2)

            # Mix two examples
            r = np.array(random.random())
            sound = U.mix(sound1, sound2, r, self.opt.fs).astype(np.float32)
            eye = np.eye(self.opt.nClasses)
            label = (eye[label1] * r + eye[label2] * (1 - r)).astype(np.float32)

        else:  # Training phase of standard learning or testing phase
            sound, label = self.base[i]
            sound = self.preprocess(sound).astype(np.float32)
            label = np.array(label, dtype=np.int32)

        if self.train and self.opt.strongAugment:
            sound = U.random_gain(6)(sound).astype(np.float32)

        return sound, label



# Test
train = ESC50(folds=[1,2,3], audio_rate=16000)
next(train.data_gen)
