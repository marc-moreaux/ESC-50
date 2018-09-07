import pandas as pd
import numpy as np
import bc_utils as U
import scipy.io.wavfile as sci_wav
import random
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
                 audio_rate=44100,
                 strongAugment=False,
                 pad=0,
                 inputLength=0,
                 random_crop=False,
                 mix=False,
                 normalize=False):
        '''Initialize the generator

        Parameters
        ----------
        only_ESC10: Bool
            Wether to use ESC10 instead of ESC50
        randomize: Bool
            Randomize samples 
        audio_rate: int
            Audio rate of our samples
        strongAugment: Bool 
           rAndom scale and put gain in audio input 
        pad: int
            Add padding before and after audio signal
        inputLength: float
            Time in seconds of the audio input
        random_crop: Bool
            Perform random crops
        normalize: int
            Value used to normalize input
        mix: Bool
            Wether to mix samples or not (between classes learning)
        '''
        self.csv_path = '../meta/esc50.csv'
        self.wav_dir = '../audio'
        self.audio_rate = audio_rate
        self.randomize = randomize
        self.audio_rate = audio_rate
        self.strongAugment = strongAugment
        self.pad = pad 
        self.inputLength = inputLength
        self.random_crop = random_crop
        self.normalize = normalize
        self.mix = mix
        self.n_classes = 50

        self.df = pd.read_csv(self.csv_path)
        self.df[self.df.fold.isin(folds)]
        if only_ESC10 is True:
            self.df = self.df[self.df['esc10']] 
            self.n_classes = 10

        self._preprocess_setup()
        self.data_gen = self._data_gen()

    def _data_gen(self):
        self.stop = False
        while not self.stop:
            idxs1 = list(self.df.index)
            idxs2 = list(self.df.index)
            if self.randomize:
                random.shuffle(idxs1)
                random.shuffle(idxs2)

            for idx1, idx2 in zip(idxs1, idxs2):
                fname1 = self.df.filename[idx1]
                fname2 = self.df.filename[idx2]
                sound1 = self.fname_to_wav(fname1)
                sound2 = self.fname_to_wav(fname2)
                sound1 = self.preprocess(sound1)
                sound2 = self.preprocess(sound2)
                label1 = self.df.target[idx1]
                label2 = self.df.target[idx2]
                if self.n_classes == 10:
                    lbl_indexes = [0, 1, 10, 11, 12, 20, 21, 38, 40, 41]
                    label1 = lbl_indexes.index(label1)
                    label2 = lbl_indexes.index(label2)

                if self.mix:  # Mix two examples
                    r = np.array(random.random())
                    sound = U.mix(sound1, sound2, r, self.audio_rate).astype(np.float32)
                    eye = np.eye(self.n_classes)
                    label = (eye[label1] * r + eye[label2] * (1 - r)).astype(np.float32)

                else:
                    sound, label = sound1, label1

                if self.strongAugment:
                    sound = U.random_gain(6)(sound).astype(np.float32)

                yield sound, label

    def batch_gen(self, batch_size):
        '''Generator yielding batches
        '''
        self.stop = False
        sounds = None
        labels = None
        data = self._data_gen()
        while not self.stop:
            for i in range(batch_size):
                if sounds is None:  # Initialize batch size
                    sound, label = next(data)
                    sounds = np.ndarray((batch_size,) + sound.shape)
                    sounds[0] = sound
                    labels = np.ndarray((batch_size,) + label.shape)
                    labels[0] = label
                else:
                    sound, label = next(data)
                    sounds[i] = sound
                    labels[i] = label

            yield (sounds, labels)

    def fname_to_wav(self, fname):
        """Retrive wav data from fname
        """
        change_audio_rate(fname, self.wav_dir, self.audio_rate)
        fpath = os.path.join(self.wav_dir, str(self.audio_rate), fname)
        wav_freq, wav_data = sci_wav.read(fpath)
        return wav_data

    def _preprocess_setup(self):
        """Apply desired pre_processing to the input
        """
        self.preprocess_funcs = []
        if self.strongAugment:
            self.preprocess_funcs.append(U.random_scale(1.25))

        if self.pad > 0:
            self.preprocess_funcs.append(U.padding(self.pad))
        
        if self.random_crop:
            self.preprocess_funcs.append(
                U.random_crop(int(self.inputLength * self.audio_rate)))

        if self.normalize is True:
            self.preprocess_funcs.append(U.normalize(32768.0))

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


def get_train_test(test_split=1, only_ESC10=False):
    '''return train and test depending on desired split
    '''
    train_splits = list(range(1,6))
    train_splits.remove(test_split)

    train = ESC50(folds=train_splits,
                  audio_rate=16000,
                  only_ESC10=only_ESC10,
                  randomize=True,
                  strongAugment=True,
                  random_crop=True,
                  pad=0,
                  inputLength=2,
                  mix=True,
                  normalize=True)

    test = ESC50(folds=[test_split],
                 audio_rate=16000,
                 only_ESC10=only_ESC10,
                 randomize=False,
                 strongAugment=False,
                 random_crop=False,
                 pad=0,
                 inputLength=4,
                 mix=False,
                 normalize=True)
    
    return train, test


def test_plot_audio():
    '''Show a train and test split
    '''
    import matplotlib.pyplot as plt
    train, test = get_train_test(test_split=1)

    fig, axs = plt.subplots(10,1)
    for i in range(10):
        sound, lbl = next(train.data_gen)
        print(sound)
        axs[i].plot(sound)

    fig, axs = plt.subplots(10,1)
    for i in range(10):
        sound, lbl = next(test.data_gen)
        print(sound)
        axs[i].plot(sound)

    plt.show()



