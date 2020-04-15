# -*- coding: utf-8 -*-
"""
Audio features used in acoss
"""
import os
import numpy as np

import essentia.standard as estd
import librosa
from essentia import Pool, array, run


__all__ = ['AudioFeatures']


class AudioFeatures(object):
    """
    Class containing methods to compute various audio features
    
    Attributes:
        hop_length: int
            Hop length between frames.  Same across all features
        fs: int
            Sample rate
        audio_file: string
            Path to audio file
        audio_vector: ndarray(N)
            List of audio samples
    
    Example use :
                >>> feature = AudioFeatures("./data/test_audio.wav")
                #chroma cens with default parameters
                >>> feature.chroma_cens()
                #chroma stft with default parameters
                >>> feature.chroma_stft()

    """

    def __init__(self, audio_file, mono=True, hop_length=512, sample_rate=44100, normalize_gain=False, verbose=False):
        """[summary]
        
        Arguments:
            audio_file {[type]} -- [description]
        
        Keyword Arguments:
            mono {bool} -- [description] (default: {True})
            hop_length {int} -- [description] (default: {512})
            sample_rate {int} -- [description] (default: {44100})
            normalize_gain {bool} -- [description] (default: {False})
            verbose {bool} -- [description] (default: {False})
        """
        self.hop_length = hop_length
        self.fs = sample_rate
        self.audio_file = audio_file
        if normalize_gain:
            if os.path.splitext(self.audio_file)[1] == ".wav":
                self.audio_vector = estd.EasyLoader(filename=audio_file, sampleRate=self.fs, replayGain=-9)()
            else:
                self.audio_vector, fs = librosa.load(audio_file, sr=self.fs, mono=True)
        elif mono:
            if os.path.splitext(self.audio_file)[1] == ".wav":
                self.audio_vector = estd.MonoLoader(filename=audio_file, sampleRate=self.fs)()
            else:
                self.audio_vector, fs = librosa.load(audio_file, sr=self.fs, mono=True)

        if verbose:
            print("== Audio vector of %s loaded with shape %s and sample rate %s =="
                  % (audio_file, self.audio_vector.shape, self.fs))

    def resample_audio(self, target_sample_rate):
        """Downsample a audio into a target sample rate
        
        Arguments:
            target_sample_rate {[type]} -- [description]
        
        Raises:
            ValueError: If `target_sample_rate` is less than the sample rate of given audio data.
        
        Returns:
            [type] -- [description]
        """
        if target_sample_rate > self.fs:
            raise ValueError("Target_sample_rate should be lower than %s" % self.fs)
        resampler = estd.Resample(inputSampleRate=self.fs, outputSampleRate=target_sample_rate, quality=1)
        self.fs = target_sample_rate
        return resampler.compute(self.audio_vector)

    def audio_slicer(self, endTime, startTime=0):
        """
        Trims the audio signal array with a specified start and end time in seconds

        Parameters
        ----------
        endTime: endTime for slicing
        startTime: (default: 0)

        Returns
        -------
        trimmed_audio: ndarray
        """
        trimmer = estd.Trimmer(startTime=startTime, endTime=endTime, checkRange=True)
        return trimmer.compute(self.audio_vector)

    def librosa_noveltyfn(self, maxSize=3, params=None):
        """
        Compute librosa's onset envelope from an input signal.

        Parameters
        ----------
        maxSize: int
            MaxSize of librosa_noveltyfn

        params: dictionary (optional)
            Dictionary with above parameters

        Returns
        -------
        novfn: ndarray(n_frames)
            Evaluation of the audio novelty function at each audio frame,
            in time increments equal to self.hop_length
        """
        if params is not None:
            if 'maxSize' in params.keys():
                maxSize = params['maxSize']

        # Include max_size=3 to make like superflux
        return librosa.onset.onset_strength(y=self.audio_vector, sr=self.fs, 
                                            hop_length=self.hop_length, max_size=maxSize)

    def madmom_features(self, fps=100, maxSize=3, nrbands=24, params=None):
        """
        Call Madmom's implementation of RNN + DBN beat tracking. Madmom's
        results are returned in terms of seconds, but round and convert to
        be in terms of hop_size so that they line up with the features.
        The novelty function is also computed as a side effect (and is
        the bottleneck in the computation), so also return that

        Parameters
        ----------
        fps: int
            Frames per second in processing

        maxSize: int
            MaxSize of librosa_noveltyfn

        nrbands: int
            Num Bands of SpectralOnsetProcessor

        params: dictionary optional
            Dictionary with above parameters

        Returns
        -------
        output: a python dict with following key, value pairs
            {
                'tempos': ndarray(n_levels, 2)
                    An array of tempo estimates in beats per minute,
                    along with their confidences
                'onsets': ndarray(n_onsets)
                    Array of onsets, where each onset indexes into a particular window
                'novfn': ndarray(n_frames)
                    Evaluation of the rnn audio novelty function at each audio
                    frame, in time increments equal to self.hop_length
                'snovfn': ndarray(n_frames)
                    Superflux audio novelty function at each audio frame,
                    in time increments equal to self.hop_length
            }
        """
        from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
        from madmom.features.tempo import TempoEstimationProcessor
        from madmom.features.onsets import SpectralOnsetProcessor
        from madmom.audio.filters import LogarithmicFilterbank

        if params is not None:
            if 'fps' in params.keys():
                fps = params['fps']
            if 'maxSize' in params.keys():
                maxSize = params['maxSize']
            if 'nrbands' in params.keys():
                nrbands = params['nrbands']

        beatproc = DBNBeatTrackingProcessor(fps=fps)
        tempoproc = TempoEstimationProcessor(fps=fps)
        tmpfile = None
        try:
            novfn = RNNBeatProcessor()(self.audio_file)  # This step is the computational bottleneck
        except:
            filepath, ext = os.path.splitext(self.audio_file)
            tmpfile = filepath + ".wav"
            signal, fs = librosa.load(self.audio_file, sr=self.fs)
            librosa.output.write_wav(tmpfile, signal, fs)
            novfn = RNNBeatProcessor()(tmpfile)

        beats = beatproc(novfn)
        tempos = tempoproc(novfn)
        onsets = np.array(np.round(beats * self.fs / float(self.hop_length)), dtype=np.int64)
        # Resample the audio novelty function to correspond to the
        # correct hop length
        nframes = len(self.librosa_noveltyfn(maxSize=maxSize))
        novfn = np.interp(np.arange(nframes) * self.hop_length / float(self.fs), np.arange(len(novfn)) / float(fps),
                          novfn)

        # For good measure, also compute and return superflux
        sodf = SpectralOnsetProcessor(onset_method='superflux',
                                      fps=fps,
                                      filterbank=LogarithmicFilterbank,
                                      num_bands=nrbands,
                                      log=np.log10)
        try:
            snovfn = sodf(self.audio_file)
        except:
            if tmpfile and os.path.exists(tmpfile):
                snovfn = sodf(tmpfile)
            else:
                filepath, ext = os.path.splitext(self.audio_file)
                tmpfile = filepath+ ".wav"
                signal, fs = librosa.load(self.audio_file, sr=self.fs)
                librosa.output.write_wav(tmpfile, signal, fs)
                snovfn = sodf(tmpfile)

        if tmpfile and os.path.exists(tmpfile):
            os.remove(tmpfile)
        snovfn = np.interp(np.arange(nframes)*self.hop_length/float(self.fs), np.arange(len(snovfn))/float(fps), snovfn) 
        return {'tempos': tempos, 'onsets': onsets, 'novfn': novfn, 'snovfn': snovfn}


    def librosa_many_onsets(self, temposbias=[60,120,180], params=None):
        """
        Call librosa's implementation of dynamic programming beat tracking
        for many bias

        Parameters
        ----------
        temposbias: list
            Bias for beat tracking

        params: dictionary (optional)
            Dictionary with above parameters

        Returns
        -------
        {
            '60': {
                    'tempo': float
                         Average tempo
                    'onsets': ndarray(n_onsets)
                        of beat intervals in number of windows
                  },

            120: {
                    'tempo': float
                         Average tempo
                    'onsets': ndarray(n_onsets)
                        of beat intervals in number of windows
                  },
            ...
         }
        """

        if params is not None:
            if 'tempobias' in params.keys():
                temposbias = params['tempobias']

        tempos = {}
        for tempobias in temposbias:
            dic_onsets = self.librosa_onsets(tempobias=tempobias)
            tempos[str(tempobias)] = dic_onsets

        return tempos


    def librosa_onsets(self, tempobias=120.0, params=None):
        """
        Call librosa's implementation of dynamic programming beat tracking

        Parameters
        ----------
        tempobias: float
            Bias for beat tracking

        params: dictionary (optional)
            Dictionary with above parameters

        Returns
        -------
        {
            'tempo': float
                 Average tempo
            'onsets': ndarray(n_onsets)
                of beat intervals in number of windows
        }
        """

        if params is not None:
            if 'tempobias' in params.keys():
                tempobias = params['tempobias']

        y_harmonic, y_percussive = librosa.effects.hpss(self.audio_vector)
        tempo, onsets = librosa.beat.beat_track(y=y_percussive, sr=self.fs, start_bpm=tempobias)
        return {'tempo': tempo, 'onsets': onsets}

    @staticmethod
    def resample_feature(feature_array, factor):
        """
        downsample a input feature array with a given step size
        """
        from scipy.signal import resample
        frames = feature_array.shape[0]
        re_size = int(np.ceil(frames / float(factor)))
        return resample(feature_array, re_size)

    def chroma_stft(self, frameSize=4096, norm=2.0, tuning = 0.0, n_chroma=12, display=False, params=None):
        """
        Computes the chromagram from the short-term fourier transform of the input audio signal

        Parameters
        ----------
        frameSize: int
            FFT window size
        tuning: float
            Deviation from A440 tuning in fractional chroma bins (librosa)
        norm: float
            Column-wise normalization (librosa)

        params: dictionary (optional)
            Dictionary with above parameters

        Returns
        -------
        chromagram: np.ndarray [shape=(t, n_chroma)]
            Normalized energy for each chroma bin at each frame.
        """

        if params is not None:
            if 'frameSize' in params.keys():
                frameSize = params['frameSize']
            if 'tuning' in params.keys():
                tuning = params['tuning']
            if 'norm' in params.keys():
                norm = params['norm']
            if 'n_chroma' in params.keys():
                n_chroma = params['n_chroma']

        chroma = librosa.feature.chroma_stft(y=self.audio_vector,
                                            sr=self.fs,
                                            tuning=tuning,
                                            norm=norm,
                                            hop_length=self.hop_length,
                                            n_fft=frameSize,
                                            n_chroma=n_chroma)
        if display:
            display_chroma(chroma, self.hop_length)
        return chroma.T

    def chroma_cqt(self, n_chroma=12,n_octaves=7,bins_per_octave=12,fmin='C1', display=False, params=None):
        """
        Computes the chromagram feature from the constant-q transform of the input audio signal
        """

        if params is not None:
            if 'fmin' in params.keys():
                if isinstance(params['fmin'], str):
                    fmin = librosa.note_to_hz(params['fmin'])
                else:
                    fmin = params['fmin']
            if 'n_chroma' in params.keys():
                n_chroma = params['n_chroma']
            if 'n_octaves' in params.keys():
                n_octaves = params['n_octaves']
            if 'bins_per_octave' in params.keys():
                bins_per_octave = params['bins_per_octave']
        else:
            if fmin and isinstance(fmin, str):
                fmin = librosa.note_to_hz(fmin)

        chroma = librosa.feature.chroma_cqt(y=self.audio_vector,
                                            sr=self.fs,
                                            hop_length=self.hop_length,
                                            fmin=fmin,
                                            n_chroma=n_chroma,
                                            n_octaves=n_octaves,
                                            bins_per_octave=bins_per_octave)
        if display:
            display_chroma(chroma, self.hop_length)
        return chroma.T

    def chroma_cens(self, n_chroma=12,n_octaves=7,bins_per_octave=12, fmin='C1', win_len_smooth=41, display=False, params=None):
        """
        Computes CENS chroma vectors for the input audio signal (numpy array)
        Refer https://librosa.github.io/librosa/generated/librosa.feature.chroma_cens.html for more parameters
        """
        if params is not None:
            if 'fmin' in params.keys():
                if isinstance(params['fmin'], str):
                    fmin = librosa.note_to_hz(params['fmin'])
                else:
                    fmin = params['fmin']
            if 'n_chroma' in params.keys():
                n_chroma = params['n_chroma']
            if 'n_octaves' in params.keys():
                n_octaves = params['n_octaves']
            if 'bins_per_octave' in params.keys():
                bins_per_octave = params['bins_per_octave']
            if  'win_len_smooth' in params.keys():
                win_len_smooth = params['win_len_smooth']
        else:
            if fmin and isinstance(fmin, str):
                fmin = librosa.note_to_hz(fmin)

        chroma_cens = librosa.feature.chroma_cens(y=self.audio_vector,
                                                  sr=self.fs,
                                                  hop_length=self.hop_length,
                                                  fmin=fmin,
                                                  n_chroma=n_chroma,
                                                  n_octaves=n_octaves,
                                                  bins_per_octave=bins_per_octave,
                                                  win_len_smooth=win_len_smooth)
        if display:
            display_chroma(chroma_cens, self.hop_length)
        return chroma_cens.T

    def chroma_cqt_processed(self, n_chroma=12,n_octaves=7,bins_per_octave=12, fmin='C1',
                             margin=8, kernel_size=31,power=2.0,mask=False,  params=None):
        """
        Adapted from librosa docs
        https://librosa.github.io/librosa_gallery/auto_examples/plot_chroma.html
        """
        from scipy.ndimage import median_filter

        if params is not None:
            if 'fmin' in params.keys():
                if isinstance(params['fmin'], str):
                    fmin = librosa.note_to_hz(params['fmin'])
                else:
                    fmin = params['fmin']
            if 'n_chroma' in params.keys():
                n_chroma = params['n_chroma']
            if 'n_octaves' in params.keys():
                n_octaves = params['n_octaves']
            if 'bins_per_octave' in params.keys():
                bins_per_octave = params['bins_per_octave']
            if  'margin' in params.keys():
                margin = params['margin']
            if 'kernel_size' in params.keys():
                kernel_size = params['kernel_size']
            if 'power' in params.keys():
                power = params['power']
            if  'mask' in params.keys():
                mask = params['mask']
        else:
            if fmin and isinstance(fmin, str):
                fmin = librosa.note_to_hz(fmin)


        harmonic = librosa.effects.harmonic(y=self.audio_vector, margin=margin,
                                            kernel_size=kernel_size,
                                            power=power,
                                            mask=mask)
        chroma_cqt_harm = librosa.feature.chroma_cqt(y=harmonic,
                                                     sr=self.fs,
                                                     bins_per_octave=bins_per_octave,
                                                     hop_length=self.hop_length,
                                                     fmin=fmin,
                                                     n_chroma=n_chroma,
                                                     n_octaves= n_octaves)
        chroma_filter = np.minimum(chroma_cqt_harm,
                           librosa.decompose.nn_filter(chroma_cqt_harm,
                                                       aggregate=np.median,
                                                       metric='cosine'))
        chroma_smooth = median_filter(chroma_filter, size=(1, 9))
        return {'chroma_filtered': chroma_filter, 
                'chroma_smoothed': chroma_smooth}

    def hpcp(self,
            frameSize=4096,
            windowType='blackmanharris62',
            harmonicsPerPeak=8,
            magnitudeThreshold=0,
            maxPeaks=100,
            whitening=True,
            referenceFrequency=440,
            minFrequency=100,
            maxFrequency=3500,
            nonLinear=False,
            numBins=12,
            display=False,
            params=None):
        """
        Compute Harmonic Pitch Class Profiles (HPCP) for the input audio files using essentia standard mode using
        the default parameters as mentioned in [1].
        Please refer to the following paper for detailed explanantion of the algorithm.
        [1]. Gómez, E. (2006). Tonal Description of Polyphonic Audio for Music Content Processing.
        For full list of parameters of essentia standard mode HPCP 
        please refer to http://essentia.upf.edu/documentation/reference/std_HPCP.html
        
        Returns
        hpcp: ndarray(n_frames, 12)
            The HPCP coefficients at each time frame
        """
        if params is not None:
            if 'frameSize' in params.keys():
                frameSize = params['frameSize']
            if 'windowType' in params.keys():
                windowType = params['windowType']
            if 'harmonicsPerPeak' in params.keys():
                harmonicsPerPeak = params['harmonicsPerPeak']
            if  'magnitudeThreshold' in params.keys():
                magnitudeThreshold = params['magnitudeThreshold']
            if 'maxPeaks' in params.keys():
                maxPeaks = params['maxPeaks']
            if 'whitening' in params.keys():
                whitening = params['whitening']
            if  'referenceFrequency' in params.keys():
                referenceFrequency = params['referenceFrequency']
            if 'minFrequency' in params.keys():
                minFrequency = params['minFrequency']
            if 'maxFrequency' in params.keys():
                maxFrequency = params['maxFrequency']
            if 'nonLinear' in params.keys():
                nonLinear = params['nonLinear']
            if 'numBins' in params.keys():
                numBins = params['numBins']


        audio = array(self.audio_vector)
        frameGenerator = estd.FrameGenerator(audio, frameSize=frameSize, hopSize=self.hop_length)
        # framecutter = estd.FrameCutter(frameSize=frameSize, hopSize=self.hop_length)
        windowing = estd.Windowing(type=windowType)
        spectrum = estd.Spectrum()
        # Refer http://essentia.upf.edu/documentation/reference/streaming_SpectralPeaks.html
        spectralPeaks = estd.SpectralPeaks(magnitudeThreshold=magnitudeThreshold,
                                            maxFrequency=maxFrequency,
                                            minFrequency=minFrequency,
                                            maxPeaks=maxPeaks,
                                            orderBy="frequency",
                                            sampleRate=self.fs)
        # http://essentia.upf.edu/documentation/reference/streaming_SpectralWhitening.html
        spectralWhitening = estd.SpectralWhitening(maxFrequency= maxFrequency,
                                                    sampleRate=self.fs)
        # http://essentia.upf.edu/documentation/reference/streaming_HPCP.html
        hpcp = estd.HPCP(sampleRate=self.fs,
                        maxFrequency=maxFrequency,
                        minFrequency=minFrequency,
                        referenceFrequency=referenceFrequency,
                        nonLinear=nonLinear,
                        harmonics=harmonicsPerPeak,
                        size=numBins)
        pool = Pool()

        #compute hpcp for each frame and add the results to the pool
        for frame in frameGenerator:
            spectrum_mag = spectrum(windowing(frame))
            frequencies, magnitudes = spectralPeaks(spectrum_mag)
            if whitening:
                w_magnitudes = spectralWhitening(spectrum_mag,
                                                frequencies,
                                                magnitudes)
                hpcp_vector = hpcp(frequencies, w_magnitudes)
            else:
                hpcp_vector = hpcp(frequencies, magnitudes)
            pool.add('tonal.hpcp',hpcp_vector)

        if display:
            display_chroma(pool['tonal.hpcp'].T, self.hop_length)

        return pool['tonal.hpcp']

    def crema(self, params=None):
        """
        Compute "convolutional and recurrent estimators for music analysis" (CREMA)
        and resample so that it's reported in hop_length intervals
        NOTE: This code is a bit finnecky, and is recommended for Python 3.5.
        Check `wrapper_cream_feature` for the actual implementation.

        Returns
        -------
        crema: ndarray(n_frames, 12)
            The crema coefficients at each frame
        """
        import crema
        from scipy import interpolate

        model = crema.models.chord.ChordModel()
        data = model.outputs(y=self.audio_vector, sr=self.fs)
        fac = (float(self.fs) / 44100.0) * 4096.0 / self.hop_length
        times_orig = fac * np.arange(len(data['chord_bass']))
        nwins = int(np.floor(float(self.audio_vector.size) / self.hop_length))
        times_new = np.arange(nwins)
        interp = interpolate.interp1d(times_orig, data['chord_pitch'].T, kind='nearest', fill_value='extrapolate')
        return interp(times_new).T

    def two_d_fft_mag(self, feature_type='chroma_cqt', display=False, params=None):
        """
        Computes 2d - fourier transform magnitude coefficients of the input feature vector (numpy array)
        Usually fed by Constant-q transform or chroma feature vectors for cover detection tasks.

        Parameters
        ----------
        feature_type: str
            Feature type to extract

        params: dictionary (optional)
            Dictionary with features parameters

        Returns
        -------
        2DFFT: np.ndarray
        """

        if feature_type == 'audio':
            feature_vector = self.audio_vector
        elif feature_type == 'hpcp':
            feature_vector = self.hpcp(params=params)
        elif feature_type == 'chroma_cqt':
            feature_vector = self.chroma_cqt(params=params)
        elif feature_type == 'chroma_cens':
            feature_vector = self.chroma_cens(params=params)
        elif feature_type == 'crema':
            feature_vector = self.crema(params=params)
        else:
            raise IOError("two_d_fft_mag: Wrong parameter 'feature type'. "
                          "Should be in one of these ['audio', 'hpcp', 'chroma_cqt', 'chroma_cens', 'crema']")

        # 2d fourier transform
        ndim_fft = np.fft.fft2(feature_vector)
        ndim_fft_mag = np.abs(np.fft.fftshift(ndim_fft))

        if display:
            import matplotlib.pyplot as plt
            from librosa.display import specshow
            plt.figure(figsize=(8,6))
            plt.title('2D-Fourier transform magnitude coefficients')
            specshow(ndim_fft_mag, cmap='jet')

        return ndim_fft_mag

    def key_extractor(self, 
                    frameSize=4096, 
                    hpcpSize=12, 
                    maxFrequency=3500,  
                    minFrequency=25, 
                    windowType='hann',
                    profileType='bgate',
                    pcpThreshold=0.2,
                    tuningFrequency=440,
                    weightType='cosine',
                    params=None):
        """
        Wrapper around essentia KeyExtractor algo. This algorithm extracts key/scale for an audio signal. 
        It computes HPCP frames for the input signal and applies key estimation using the Key algorithm.

        Refer for more details https://essentia.upf.edu/documentation/reference/streaming_KeyExtractor.html

        Returns:
                a dictionary with corresponding values for key, scale and strength

        eg: {'key': 'F', 'scale': 'major', 'strength': 0.7704258561134338}

        """
        if params is not None:
            if 'frameSize' in params.keys():
                frameSize = params['frameSize']
            if 'tuningFrequency' in params.keys():
                tuningFrequency = params['tuningFrequency']
            if 'hpcpSize' in params.keys():
                hpcpSize = params['hpcpSize']
            if 'maxFrequency' in params.keys():
                maxFrequency = params['maxFrequency']
            if 'minFrequency' in params.keys():
                minFrequency = params['minFrequency']
            if 'windowType' in params.keys():
                windowType = params['windowType']
            if 'profileType' in params.keys():
                profileType = params['profileType']
            if 'pcpThreshold' in params.keys():
                pcpThreshold = params['pcpThreshold']
            if 'weightType' in params.keys():
                weightType = params['weightType']


        audio = array(self.audio_vector)
        key = estd.KeyExtractor(frameSize=frameSize, hopSize=self.hop_length, tuningFrequency=tuningFrequency)
        """
        TODO: test it with new essentia update
        key = ess.KeyExtractor(frameSize=frameSize,
                               hopSize=self.hop_length,
                               sampleRate=self.fs, 
                               hpcpSize=hpcpSize,
                               maxFrequency=maxFrequency,
                               minFrequency=minFrequency,
                               windowType=windowType,
                               profileType=profileType,
                               pcpThreshold=pcpThreshold,
                               tuningFrequency=tuningFrequency,
                               weightType=weightType)
        """
        key, scale, strength = key.compute(audio)

        return {'key': key, 'scale': scale, 'strength': strength}

    def tempogram(self, win_length=384, center=True, window='hann', params=None):
        """
        Compute the tempogram: local autocorrelation of the onset strength envelope. [1]
        [1] Grosche, Peter, Meinard Müller, and Frank Kurth. “Cyclic tempogram - A mid-level tempo
        representation for music signals.” ICASSP, 2010.

        https://librosa.github.io/librosa/generated/librosa.feature.tempogram.html
        """
        if params is not None:
            if 'win_length' in params.keys():
                win_length = params['win_length']
            if 'center' in params.keys():
                center = params['center']
            if 'window' in params.keys():
                window = params['window']

        return librosa.feature.tempogram(y=self.audio_vector,
                                         sr=self.fs,
                                         onset_envelope=self.librosa_noveltyfn(),
                                         hop_length=self.hop_length,
                                         win_length=win_length,
                                         center=center,
                                         window=window)

    def cqt_nsg(self, frame_size=4096,
                    transitionSize= 1024,
                    minFrequency= 65.41,
                    maxFrequency= 6000,
                    binsPerOctave= 48,
                    rasterize= 'full',
                    phaseMode= 'global',
                    gamma= 0,
                    normalize= 'none',
                    window= 'hannnsgcq',
                    params=None):
        """
        invertible CQT algorithm based on Non-Stationary Gabor frames
        https://mtg.github.io/essentia-labs//news/2019/02/07/invertible-constant-q/
        https://essentia.upf.edu/documentation/reference/std_NSGConstantQ.html
        """
        import essentia.pytools.spectral as epy

        if params is not None:
            if 'frame_size' in params.keys():
                frame_size = params['frame_size']
            if 'transitionSize' in params.keys():
                transitionSize = params['transitionSize']
            if 'minFrequency' in params.keys():
                minFrequency = params['minFrequency']
            if 'maxFrequency' in params.keys():
                maxFrequency = params['maxFrequency']
            if 'binsPerOctave' in params.keys():
                binsPerOctave = params['binsPerOctave']
            if 'rasterize' in params.keys():
                rasterize = params['rasterize']
            if 'phaseMode' in params.keys():
                phaseMode = params['phaseMode']
            if 'gamma' in params.keys():
                gamma = params['gamma']
            if 'normalize' in params.keys():
                normalize = params['normalize']
            if 'window' in params.keys():
                window = params['window']


        cq_frames, dc_frames, nb_frames = epy.nsgcqgram(self.audio_vector, sampleRate=self.fs,
                                                        frameSize=frame_size,
                                                        transitionSize=transitionSize,
                                                        minFrequency=minFrequency,
                                                        maxFrequency=maxFrequency,
                                                        binsPerOctave=binsPerOctave,
                                                        rasterize=rasterize,
                                                        phaseMode=phaseMode,
                                                        gamma=gamma,
                                                        normalize=normalize,
                                                        window=window)
        return cq_frames

    def cqt(self, fmin=None, n_bins=84, bins_per_octave=12, tuning=0.0,
              filter_scale=1, norm=1, sparsity=0.01, window='hann', scale=True, pad_mode='reflect',params=None):
        """
        Compute the constant-Q transform implementation as in librosa
        https://librosa.github.io/librosa/generated/librosa.core.cqt.html
        """

        if params is not None:
            if 'fmin' in params.keys():
                fmin = params['fmin']
            if 'n_bins' in params.keys():
                n_bins = params['n_bins']
            if 'bins_per_octave' in params.keys():
                bins_per_octave = params['bins_per_octave']
            if 'tuning' in params.keys():
                tuning = params['tuning']
            if 'filter_scale' in params.keys():
                filter_scale = params['filter_scale']
            if 'norm' in params.keys():
                norm = params['norm']
            if 'sparsity' in params.keys():
                sparsity = params['sparsity']
            if 'window' in params.keys():
                window = params['window']
            if 'scale' in params.keys():
                scale = params['scale']
            if 'pad_mode' in params.keys():
                pad_mode = params['pad_mode']


        return librosa.core.cqt(y=self.audio_vector,
                                sr=self.fs,
                                hop_length=self.hop_length,
                                fmin=fmin,
                                n_bins=n_bins,
                                bins_per_octave=bins_per_octave,
                                tuning=tuning,
                                filter_scale=filter_scale,
                                norm=norm,
                                sparsity=sparsity,
                                window=window,
                                scale=scale,
                                pad_mode=pad_mode)

    def mfcc_htk(self, windowType= 'hamming',
                window_length= 22050,
                type= 'magnitude',  # htk uses mel filterbank magniude
                warpingFormula= 'htkMel',  # htk's mel warping formula
                weighting= 'linear',  # computation of filter weights done in Hz domain
                highFrequencyBound= 8000,  # 8000 is htk default
                lowFrequencyBound= 0,  # corresponds to htk default
                numberBands= 26,  # corresponds to htk default  NUMCHANS = 26
                numberCoefficients= 13,
                normalize= 'unit_max',  # htk filter normaliation to have constant height = 1
                dctType= 3,  # htk uses DCT type III
                logType= 'log',
                liftering= 22,
                params=None):
        """
        Get MFCCs 'the HTK way' with the help of Essentia
        https://github.com/MTG/essentia/blob/master/src/examples/tutorial/example_mfcc_the_htk_way.py
        Using all of the default parameters from there except the hop length (which shouldn't matter), and a much longer window length (which has been found to work better for covers)
        Parameters
        ----------
        window_length: int
            Length of the window to use for the STFT
        nmfcc: int
            Number of MFCC coefficients to compute
        n_mels: int
            Number of frequency bands to use
        fmax: int
            Maximum frequency
        Returns
        -------
        ndarray(nmfcc, nframes)
            An array of all of the MFCC frames
        """

        if params is not None:
            if 'windowType' in params.keys():
                windowType = params['windowType']
            if 'window_length' in params.keys():
                window_length = params['window_length']
            if 'type' in params.keys():
                type = params['type']
            if 'warpingFormula' in params.keys():
                warpingFormula = params['warpingFormula']
            if 'weighting' in params.keys():
                weighting = params['weighting']
            if 'highFrequencyBound' in params.keys():
                highFrequencyBound = params['highFrequencyBound']
            if 'lowFrequencyBound' in params.keys():
                lowFrequencyBound = params['lowFrequencyBound']
            if 'numberBands' in params.keys():
                numberBands = params['numberBands']
            if 'numberCoefficients' in params.keys():
                numberCoefficients = params['numberCoefficients']
            if 'normalize' in params.keys():
                normalize = params['normalize']
            if 'dctType' in params.keys():
                dctType = params['dctType']
            if 'logType' in params.keys():
                logType = params['logType']
            if 'liftering' in params.keys():
                liftering = params['liftering']



        fftlen = int(2**(np.ceil(np.log(window_length)/np.log(2))))
        spectrumSize= fftlen//2+1
        zeroPadding = fftlen - window_length

        w = estd.Windowing(type = windowType, #  corresponds to htk default  USEHAMMING = T
                            size = window_length,
                            zeroPadding = zeroPadding,
                            normalized = False,
                            zeroPhase = False)
        
        spectrum = estd.Spectrum(size=fftlen)
        mfcc_htk = estd.MFCC(inputSize = spectrumSize,
                             type=type,  # htk uses mel filterbank magniude
                             warpingFormula=warpingFormula,  # htk's mel warping formula
                             weighting=weighting,  # computation of filter weights done in Hz domain
                             highFrequencyBound=highFrequencyBound,  # 8000 is htk default
                             lowFrequencyBound=lowFrequencyBound,  # corresponds to htk default
                             numberBands=numberBands,  # corresponds to htk default  NUMCHANS = 26
                             numberCoefficients=numberCoefficients,
                             normalize=normalize,  # htk filter normaliation to have constant height = 1
                             dctType=dctType,  # htk uses DCT type III
                             logType=logType,
                             liftering=liftering)  # corresponds to htk default CEPLIFTER = 22


        mfccs = []
        # startFromZero = True, validFrameThresholdRatio = 1 : the way htk computes windows
        for frame in estd.FrameGenerator(self.audio_vector, frameSize = window_length, hopSize = self.hop_length , startFromZero = True, validFrameThresholdRatio = 1):
            spect = spectrum(w(frame))
            mel_bands, mfcc_coeffs = mfcc_htk(spect)
            mfccs.append(mfcc_coeffs)
        
        return np.array(mfccs, dtype=np.float32).T
    
    def mfcc_librosa(self, win_length=22050,
                         window= 'hann',
                         center= True,
                         n_mels= 40,
                         fmin= 0.0,
                         fmax= 8000,
                         htk= False,
                         norm= 'slaney',
                         nmfcc= 20,
                         lifterexp= 0.6,
                         params=None):
        """
        Using the default parameters from C Tralie
        "Early MFCC And HPCP Fusion for Robust Cover Song Identification"
        Parameters
        ----------
        window_length: int
            Length of the window to use for the STFT
        nmfcc: int
            Number of MFCC coefficients to compute
        n_mels: int
            Number of frequency bands to use
        fmax: int
            Maximum frequency
        lifterexp: float
            Liftering exponent
        Returns
        -------
        ndarray(nmfcc, nframes)
            An array of all of the MFCC frames
        """

        if params is not None:
            if 'window' in params.keys():
                window = params['window']
            if 'win_length' in params.keys():
                win_length = params['win_length']
            if 'center' in params.keys():
                center = params['center']
            if 'n_mels' in params.keys():
                n_mels = params['n_mels']
            if 'fmin' in params.keys():
                fmin = params['fmin']
            if 'fmax' in params.keys():
                fmax = params['fmax']
            if 'htk' in params.keys():
                htk = params['htk']
            if 'norm' in params.keys():
                norm = params['norm']
            if 'nmfcc' in params.keys():
                nmfcc = params['nmfcc']
            if 'lifterexp' in params.keys():
                lifterexp = params['lifterexp']



        S = librosa.core.stft(self.audio_vector, win_length, self.hop_length)
        M = librosa.filters.mel(self.fs, win_length, n_mels = n_mels,
                                fmin=fmin, fmax = fmax, htk=htk, norm=norm)
        X = M.dot(np.abs(S))
        X = librosa.core.amplitude_to_db(X)
        X = np.dot(librosa.filters.dct(nmfcc, X.shape[0]), X) #Make MFCC
        #Do liftering
        coeffs = np.arange(nmfcc)**lifterexp
        coeffs[0] = 1
        X = coeffs[:, None]*X
        X = np.array(X, dtype = np.float32)
        return X

    def export_onset_clicks(self, outname, onsets):
        """
        Test a beat tracker by creating an audio file
        with little blips where the onsets are
        Parameters
        ----------
        outname: string 
            Path to the file to which to output
        onsets: ndarray(n_onsets)
            An array of onsets, in terms of the hop length
        """
        import scipy.io as sio
        import subprocess
        yaudio = np.array(self.audio_vector)
        blipsamples = int(np.round(0.02*self.fs))
        blip = np.cos(2*np.pi*np.arange(blipsamples)*440.0/self.fs)
        blip = np.array(blip*np.max(np.abs(yaudio)), dtype=yaudio.dtype)
        for idx in onsets:
            l = len(yaudio[idx*self.hop_length:idx*self.hop_length+blipsamples])
            yaudio[idx*self.hop_length:idx*self.hop_length+blipsamples] = blip[0:l]
        sio.wavfile.write("temp.wav", self.fs, yaudio)
        if os.path.exists(outname):
            os.remove(outname)
        subprocess.call(["ffmpeg", "-i", "temp.wav", outname])
        os.remove("temp.wav")

    def chromaprint(self, analysisTime=30):
        """
        This algorithm computes the fingerprint of the input signal using Chromaprint algorithm. 
        It is a wrapper of the Chromaprint library

        Returns: The chromaprints are returned as base64-encoded strings.
        """
        import essentia.streaming as ess

        vec_input = ess.VectorInput(self.audio_vector)
        chromaprinter = ess.Chromaprinter(analysisTime=analysisTime, sampleRate=self.fs)
        pool = Pool()

        vec_input.data >> chromaprinter.signal
        chromaprinter.fingerprint >> (pool, 'chromaprint')
        run(vec_input)
        return pool['chromaprint']


def display_chroma(chroma, hop_length=512, fs=44100):
    """
    Make plots for input chroma vector using librosa's spechow
    Parameters
    ----------
    chroma: ndarray(n_frames, n_chroma_bins)
        An array of chroma features
    """
    from librosa.display import specshow
    specshow(chroma.T, x_axis='time', y_axis='chroma', hop_length=hop_length, sr=fs)
