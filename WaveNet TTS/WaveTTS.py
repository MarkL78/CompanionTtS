import numpy as np
import librosa
from keras.models import Model
from keras.layers import Input, Conv1D, Activation, Dropout, Add
from keras.layers import Dense, Multiply, Lambda, Reshape

