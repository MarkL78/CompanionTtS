import numpy as np
import librosa
from keras.models import Model
from keras.layers import Input, Conv1D, Activation, Dropout, Add
from keras.layers import Dense, Multiply, Lambda, Reshape

# Audio File used to test the quality of the voice
filename = 'audio.wav'
x, sr = librosa.load(filename, sr=16000)

# Function that will prepare the data that the model will be trained on
def prepare_data(data, receptive_field):
    # zero padding
    data = np.pad(data, (receptive_field, 0), 'constant', constant_values=0)
    # make training data and target data
    train_data = np.zeros((data.shape[0]-receptive_field, receptive_field))
    target_data = np.zeros((data.shape[0]-receptive_field, 1))
    for i in range(data.shape[0]-receptive_field):
        train_data[i] = data[i:i+receptive_field]
        target_data[i] = data[i+receptive_field]
    train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1], 1))
    return train_data, target_data


def build_model(receptive_field):
    inputs = Input(shape=(receptive_field, 1))
    x = inputs
    skips = []
    for i in range(10):
        dilation_rate = 2 ** i
        x = Conv1D(filters=32, kernel_size=2, dilation_rate=dilation_rate, padding='causal')(x)
        x = Activation('relu')(x)
        x = Dropout(rate=0.05)(x)
        # skip connection
        s = Conv1D(filters=32, kernel_size=1)(x)
        skips.append(s)
        x = Conv1D(filters=32, kernel_size=1)(x)
        # gated activation unit
        x = Multiply()([Activation('tanh')(x), Activation('sigmoid')(x)])
        x = Multiply()([x, Lambda(lambda x: np.sqrt(1/32))(x)])
        # residual connection
        x = Add()([x, inputs])
    x = Add()(skips)
    x = Activation('relu')(x)
    x = Dropout(rate=0.05)(x)
    x = Conv1D(filters=1, kernel_size=1)(x)
    x = Reshape((-1,))(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


# Prepare data, create the model, train the model
train_data, target_data = prepare_data(x, receptive_field=512)
model = build_model(receptive_field=512)