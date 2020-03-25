PROFILE = {
           'sample_rate': 44100,
           'hop_length':512,
           'normalize_gain': False,
           'input_audio_format': '.mp3',
           'downsample_audio': False,
           'downsample_factor': 2,
           'endtime': None,
           'features': ['hpcp',
                        'key_extractor',
                        'madmom_features',
                        'mfcc_htk'],
            'madmom_features': {},
            'librosa_onsets':{'tempobias':120},
            'librosa_many_onsets':{'tempobias':[60,120,180]},
            'chroma_stft':{'tuning':0,
                           'norm':2,
                           'n_fft': 4096,
                           'n_chroma':12
                          },
            'chroma_cqt':{ 'fmin': 'C1',
                           'n_chroma':12,
                           'n_octaves':7,
                           'bins_per_octave':12
                         },
            'chroma_cens':{'fmin':'C1',
                           'n_chroma':12,
                           'n_octaves':7,
                           'bins_per_octave':12,
                           'win_len_smooth':41
                          },
            'chroma_cqt_processed': {'margin': 8,
                                     'kernel_size': 31,
                                     'power': 2.0,
                                     'mask': False,
                                     'bins_per_octave': 12 * 3,
                                     'fmin': None,
                                     'n_chroma': 12,
                                     'n_octaves': 7
                                     },
            'hpcp': {'windowType': 'blackmanharris62',
                     'magnitudeThreshold': 0,
                     'frameSize': 4096,
                     'maxFrequency': 3500,
                     'minFrequency': 100,
                     'maxPeaks': 100,
                     'orderBy': "frequency",
                     'referenceFrequency': 440,
                     'nonLinear': False,
                     'harmonics': 8,
                     'numBins': 12,
                    },
            'crema': {},
            'key_extractor': {'frameSize': 4096,
                             'hpcpSize': 12,
                             'maxFrequency': 3500,
                             'minFrequency': 25,
                             'windowType': 'hann',
                             'profileType': 'bgate',
                             'pcpThreshold': 0.2,
                             'tuningFrequency': 440,
                             'weightType': 'cosine'
                            },
            'tempogram': {'win_length': 384,
                          'center': True,
                          'window': 'hann',
                         },
            'cqt_nsg': {'frameSize': 4096,
                        'transitionSize': 1024,
                        'minFrequency': 65.41,
                        'maxFrequency': 6000,
                        'binsPerOctave': 48,
                        'rasterize': 'full',
                        'phaseMode': 'global',
                        'gamma': 0,
                        'normalize': 'none',
                        'window': 'hannnsgcq'
                       },
            'cqt': {'fmin':None,
                    'n_bins':84,
                    'bins_per_octave': 12,
                    'tuning':0.0,
                    'filter_scale': 1,
                    'norm': 1,
                    'sparsity': 0.01,
                    'window': 'hann',
                    'scale': True,
                    'pad_mode': 'reflect'
                   },
            'mfcc_htk': {'windowType': 'hamming',
                         'window_length': 22050,
                         'type': 'magnitude', # htk uses mel filterbank magniude
                         'warpingFormula': 'htkMel', # htk's mel warping formula
                         'weighting': 'linear', # computation of filter weights done in Hz domain
                         'highFrequencyBound': 8000, # 8000 is htk default
                         'lowFrequencyBound': 0, # corresponds to htk default
                         'numberBands': 26, # corresponds to htk default  NUMCHANS = 26
                         'numberCoefficients': 13,
                         'normalize': 'unit_max', # htk filter normaliation to have constant height = 1
                         'dctType': 3, # htk uses DCT type III
                         'logType': 'log',
                         'liftering': 22
                        },
            'mfcc_librosa': {'n_fft': 2048,
                             'win_length':22050,
                             'window': 'hann',
                             'center': True,
                             'n_mels': 40,
                             'fmin': 0.0,
                             'fmax': 8000,
                             'htk': False,
                             'norm': 'slaney',
                             'nmfcc': 20,
                             'lifterexp': 0.6,
                            },
          }

PROFILE_11kHz = {
           'sample_rate': 11025, #44100,
           'hop_length':128, #512, 1/4 de 44100
           'normalize_gain': False,
           'input_audio_format': '.ogg',
           'downsample_audio': False,
           'downsample_factor': None,
           'endtime': None,
           'features': ['hpcp',
                        'key_extractor',
                        'madmom_features',
                        'librosa_many_onsets',
                        'mfcc_htk',
                        'crema',
                        'chroma_cens'],
            'madmom_features': {},
            'librosa_onsets':{'tempobias':120},
            'librosa_many_onsets':{'tempobias':[60,120,180]},
            'chroma_stft':{'tuning':0,
                           'norm':2,
                           'n_fft': 1024,
                           'n_chroma':12
                          },
            'chroma_cqt':{ 'fmin': 'C1',
                           'n_chroma':12,
                           'n_octaves':7,
                           'bins_per_octave':12
                         },
            'chroma_cens':{'fmin':'C1',
                           'n_chroma':12,
                           'n_octaves':7,
                           'bins_per_octave':12,
                           'win_len_smooth':41
                          },
            'chroma_cqt_processed': {'margin': 8,
                                     'kernel_size': 31,
                                     'power': 2.0,
                                     'mask': False,
                                     'bins_per_octave': 12 * 3,
                                     'fmin': None,
                                     'n_chroma': 12,
                                     'n_octaves': 7
                                     },
            'hpcp': {'windowType': 'blackmanharris62',
                     'magnitudeThreshold': 0,
                     'frameSize': 1024,
                     'maxFrequency': 3500,
                     'minFrequency': 100,
                     'maxPeaks': 100,
                     'orderBy': "frequency",
                     'referenceFrequency': 440,
                     'nonLinear': False,
                     'harmonics': 8,
                     'numBins': 12,
                    },
            'crema': {},
            'key_extractor': {'frameSize': 1024,
                             'hpcpSize': 12,
                             'maxFrequency': 3500,
                             'minFrequency': 25,
                             'windowType': 'hann',
                             'profileType': 'bgate',
                             'pcpThreshold': 0.2,
                             'tuningFrequency': 440,
                             'weightType': 'cosine'
                            },
            'tempogram': {'win_length': 96,
                          'center': True,
                          'window': 'hann',
                         },
            'cqt_nsg': {'frameSize': 1024,
                        'transitionSize': 256,
                        'minFrequency': 65.41,
                        'maxFrequency': 5000,
                        'binsPerOctave': 48,
                        'rasterize': 'full',
                        'phaseMode': 'global',
                        'gamma': 0,
                        'normalize': 'none',
                        'window': 'hannnsgcq'
                       },
            'cqt': {'fmin':None,
                    'n_bins':84,
                    'bins_per_octave': 12,
                    'tuning':0.0,
                    'filter_scale': 1,
                    'norm': 1,
                    'sparsity': 0.01,
                    'window': 'hann',
                    'scale': True,
                    'pad_mode': 'reflect'
                   },
            'mfcc_htk': {'windowType': 'hamming',
                         'window_length': 5500,
                         'type': 'magnitude', # htk uses mel filterbank magniude
                         'warpingFormula': 'htkMel', # htk's mel warping formula
                         'weighting': 'linear', # computation of filter weights done in Hz domain
                         'highFrequencyBound': 5500, # 8000 is htk default
                         'lowFrequencyBound': 0, # corresponds to htk default
                         'numberBands': 40, # corresponds to htk default  NUMCHANS = 26
                         'numberCoefficients': 20,
                         'normalize': 'unit_max', # htk filter normaliation to have constant height = 1
                         'dctType': 3, # htk uses DCT type III
                         'logType': 'log',
                         'liftering': 22
                        },
            'mfcc_librosa': {'win_length':5500,
                             'window': 'hann',
                             'center': True,
                             'n_mels': 40,
                             'fmin': 0.0,
                             'fmax': 5500,
                             'htk': False,
                             'norm': 'slaney',
                             'nmfcc': 20,
                             'lifterexp': 0.6,
                            },
          }


PROFILE_16kHz = {
           'sample_rate': 16000,
           'hop_length':512,
           'normalize_gain': False,
           'input_audio_format': '.mp3',
           'downsample_audio': False,
           'downsample_factor': 2,
           'endtime': None,
           'features': ['hpcp',
                        'key_extractor',
                        'madmom_features',
                        'mfcc_htk'],
            'madmom_features': {},
            'librosa_onsets':{'tempobias':120},
            'librosa_many_onsets':{'tempobias':[60,120,180]},
            'chroma_stft':{'tuning':0,
                           'norm':2,
                           'n_fft': 4096,
                           'n_chroma':12
                          },
            'chroma_cqt':{ 'fmin': 'C1',
                           'n_chroma':12,
                           'n_octaves':7,
                           'bins_per_octave':12
                         },
            'chroma_cens':{'fmin':'C1',
                           'n_chroma':12,
                           'n_octaves':7,
                           'bins_per_octave':12,
                           'win_len_smooth':41
                          },
            'chroma_cqt_processed': {'margin': 8,
                                     'kernel_size': 31,
                                     'power': 2.0,
                                     'mask': False,
                                     'bins_per_octave': 12 * 3,
                                     'fmin': None,
                                     'n_chroma': 12,
                                     'n_octaves': 7
                                     },
            'hpcp': {'windowType': 'blackmanharris62',
                     'magnitudeThreshold': 0,
                     'frameSize': 4096,
                     'maxFrequency': 3500,
                     'minFrequency': 100,
                     'maxPeaks': 100,
                     'orderBy': "frequency",
                     'referenceFrequency': 440,
                     'nonLinear': False,
                     'harmonics': 8,
                     'numBins': 12,
                    },
            'crema': {},
            'key_extractor': {'frameSize': 4096,
                             'hpcpSize': 12,
                             'maxFrequency': 3500,
                             'minFrequency': 25,
                             'windowType': 'hann',
                             'profileType': 'bgate',
                             'pcpThreshold': 0.2,
                             'tuningFrequency': 440,
                             'weightType': 'cosine'
                            },
            'tempogram': {'win_length': 384,
                          'center': True,
                          'window': 'hann',
                         },
            'cqt_nsg': {'frameSize': 4096,
                        'transitionSize': 1024,
                        'minFrequency': 65.41,
                        'maxFrequency': 6000,
                        'binsPerOctave': 48,
                        'rasterize': 'full',
                        'phaseMode': 'global',
                        'gamma': 0,
                        'normalize': 'none',
                        'window': 'hannnsgcq'
                       },
            'cqt': {'fmin':None,
                    'n_bins':84,
                    'bins_per_octave': 12,
                    'tuning':0.0,
                    'filter_scale': 1,
                    'norm': 1,
                    'sparsity': 0.01,
                    'window': 'hann',
                    'scale': True,
                    'pad_mode': 'reflect'
                   },
            'mfcc_htk': {'windowType': 'hamming',
                         'window_length': 22050,
                         'type': 'magnitude', # htk uses mel filterbank magniude
                         'warpingFormula': 'htkMel', # htk's mel warping formula
                         'weighting': 'linear', # computation of filter weights done in Hz domain
                         'highFrequencyBound': 8000, # 8000 is htk default
                         'lowFrequencyBound': 0, # corresponds to htk default
                         'numberBands': 26, # corresponds to htk default  NUMCHANS = 26
                         'numberCoefficients': 13,
                         'normalize': 'unit_max', # htk filter normaliation to have constant height = 1
                         'dctType': 3, # htk uses DCT type III
                         'logType': 'log',
                         'liftering': 22
                        },
            'mfcc_librosa': {'n_fft': 2048,
                             'win_length':22050,
                             'window': 'hann',
                             'center': True,
                             'n_mels': 40,
                             'fmin': 0.0,
                             'fmax': 8000,
                             'htk': False,
                             'norm': 'slaney',
                             'nmfcc': 20,
                             'lifterexp': 0.6,
                            },
          }


PROFILE_8kHz = {
           'sample_rate': 16000, #44100,
           'hop_length':256, #512,
           'normalize_gain': False,
           'input_audio_format': '.mp3',
           'downsample_audio': 2,
           'downsample_factor': None,
           'endtime': None,
           'features': ['hpcp',
                        'key_extractor',
                        'madmom_features',
                        'librosa_many_onsets',
                        'mfcc_htk',
                        'crema',
                        'chroma_cens'],
            'madmom_features': {},
            'librosa_onsets':{'tempobias':120},
            'librosa_many_onsets':{'tempobias':[60,120,180]},
            'chroma_stft':{'tuning':0,
                           'norm':2,
                           'n_fft': 4096,
                           'n_chroma':12
                          },
            'chroma_cqt':{ 'fmin': 'C1',
                           'n_chroma':12,
                           'n_octaves':7,
                           'bins_per_octave':12
                         },
            'chroma_cens':{'fmin':'C1',
                           'n_chroma':12,
                           'n_octaves':7,
                           'bins_per_octave':12,
                           'win_len_smooth':41
                          },
            'chroma_cqt_processed': {'margin': 8,
                                     'kernel_size': 31,
                                     'power': 2.0,
                                     'mask': False,
                                     'bins_per_octave': 12 * 3,
                                     'fmin': None,
                                     'n_chroma': 12,
                                     'n_octaves': 7
                                     },
            'hpcp': {'windowType': 'blackmanharris62',
                     'magnitudeThreshold': 0,
                     'frameSize': 2048,
                     'maxFrequency': 3500,
                     'minFrequency': 100,
                     'maxPeaks': 100,
                     'orderBy': "frequency",
                     'referenceFrequency': 440,
                     'nonLinear': False,
                     'harmonics': 8,
                     'numBins': 12,
                    },
            'crema': {},
            'key_extractor': {'frameSize': 2048,
                             'hpcpSize': 12,
                             'maxFrequency': 3500,
                             'minFrequency': 25,
                             'windowType': 'hann',
                             'profileType': 'bgate',
                             'pcpThreshold': 0.2,
                             'tuningFrequency': 440,
                             'weightType': 'cosine'
                            },
            'tempogram': {'win_length': 384,
                          'center': True,
                          'window': 'hann',
                         },
            'cqt_nsg': {'frameSize': 4096,
                        'transitionSize': 1024,
                        'minFrequency': 65.41,
                        'maxFrequency': 6000,
                        'binsPerOctave': 48,
                        'rasterize': 'full',
                        'phaseMode': 'global',
                        'gamma': 0,
                        'normalize': 'none',
                        'window': 'hannnsgcq'
                       },
            'cqt': {'fmin':None,
                    'n_bins':84,
                    'bins_per_octave': 12,
                    'tuning':0.0,
                    'filter_scale': 1,
                    'norm': 1,
                    'sparsity': 0.01,
                    'window': 'hann',
                    'scale': True,
                    'pad_mode': 'reflect'
                   },
            'mfcc_htk': {'windowType': 'hamming',
                         'window_length': 4000,
                         'type': 'magnitude', # htk uses mel filterbank magniude
                         'warpingFormula': 'htkMel', # htk's mel warping formula
                         'weighting': 'linear', # computation of filter weights done in Hz domain
                         'highFrequencyBound': 4000, # 8000 is htk default
                         'lowFrequencyBound': 0, # corresponds to htk default
                         'numberBands': 26, # corresponds to htk default  NUMCHANS = 26
                         'numberCoefficients': 13,
                         'normalize': 'unit_max', # htk filter normaliation to have constant height = 1
                         'dctType': 3, # htk uses DCT type III
                         'logType': 'log',
                         'liftering': 22
                        },
            'mfcc_librosa': {'n_fft': 2048,
                             'win_length':22050,
                             'window': 'hann',
                             'center': True,
                             'n_mels': 40,
                             'fmin': 0.0,
                             'fmax': 8000,
                             'htk': False,
                             'norm': 'slaney',
                             'nmfcc': 20,
                             'lifterexp': 0.6,
                            },
          }


