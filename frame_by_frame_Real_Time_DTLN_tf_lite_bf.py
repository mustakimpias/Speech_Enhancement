
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
from beamformer import Frame_by_frame_BF as fbf_mvdr
from beamformer import util
from tensorflow.lite.python import interpreter as tflite
import tensorflow as tf

from tqdm import tqdm
import warnings

import time
warnings.filterwarnings("ignore")




print('Microphone data reading and convert to 16KHz.....')
FFT_LENGTH = 128
FFT_SHIFT = 32
block_len = 512
block_shift = 128
CHANNEL_INDEX = [1,2,3]
fs=16000

beamformer_maker = fbf_mvdr.frame_by_frame_mvdr(fs,
                                                    FFT_LENGTH,
                                                    FFT_SHIFT,
                                                    len(CHANNEL_INDEX))
beamformer = np.ones((len(CHANNEL_INDEX), FFT_LENGTH // 2 + 1), dtype=np.complex64)

sig1 = sf.read('16k3micRecordingsFixedSpeechLevelALL/mainMicNoisyMTR6.wav',dtype='float32')[0]
sig2 = sf.read('16k3micRecordingsFixedSpeechLevelALL/inEarMicNoisyMTR6.wav',dtype='float32')[0]
sig3 = sf.read('16k3micRecordingsFixedSpeechLevelALL/noiseMicNoisyMTR6.wav',dtype='float32')[0]
clean = sf.read('16k3micRecordingsFixedSpeechLevelALL/mainMicSpeech.wav', dtype='float32')[0]
noisy = sig1 + sig2 + sig3
ll = min(len(sig1),len(sig2),len(sig3),len(clean),len(noisy))

print('Total Signal duration',ll//fs ,'s' )
#print('How many seconds would you like to process in each iteration?')
#ss = 1 # int(input())
#sft = ss*(ll//(ll//fs))

#print(sft//block_shift)
#ll = sft*(ll//sft)
sig1=sig1[0:ll]
sig2=sig2[0:ll]
sig3=sig3[0:ll]
clean = clean[0:ll]
noisy = noisy[0:ll]
audio = sig1 + sig2 + sig3

multi_channel = np.zeros(shape=(ll,3),dtype='float32')
multi_channel[:,0]=sig1
multi_channel[:,1]=sig2
multi_channel[:,2]=sig3
print(audio.shape)



##########################
# the values are fixed, if you need other values, you have to retrain.
# The sampling rate of 16k is also fix.

# load models
interpreter_1 = tflite.Interpreter(model_path='./pretrained_model/model_1.tflite')
interpreter_1.allocate_tensors()
interpreter_2 = tflite.Interpreter(model_path='./pretrained_model/model_2.tflite')
interpreter_2.allocate_tensors()

# Get input and output tensors.
input_details_1 = interpreter_1.get_input_details()
output_details_1 = interpreter_1.get_output_details()

input_details_2 = interpreter_2.get_input_details()
output_details_2 = interpreter_2.get_output_details()
# create states for the lstms
states_1 = np.zeros(input_details_1[1]['shape']).astype('float32')
states_2 = np.zeros(input_details_2[1]['shape']).astype('float32')


#sf.write('output/dump_audio.wav' ,multi_channel ,fs)
# check for sampling rate
if fs != 16000:
    raise ValueError('This model only supports 16k sampling rate.')
# preallocate output audio
out_file = np.zeros((len(audio)))
# create buffer
in_buffer = np.zeros((block_len)).astype('float32')
out_buffer = np.zeros((block_len)).astype('float32')
# create buffer
in1_buffer = np.zeros((block_len)).astype('float32')
in2_buffer = np.zeros((block_len)).astype('float32')
in3_buffer = np.zeros((block_len)).astype('float32')
# calculate number of blocks
num_blocks = ll//block_shift
time_array = []
# iterate over the number of blcoks
sp_mask = []
n_mask = []
print('If you would like to performing beamforming then press 1 otherwise press 0.')
bf = int(input())
###print('1st 10 samples of input signal',audio[0:10])
print('Frequency domain enhanced signal and mask estimation....')
for idx in tqdm(range(num_blocks)):
    #print(idx)
    start_time = time.time()
    # shift values and write to buffer
    in1_buffer[:-block_shift] = in1_buffer[block_shift:]
    in1_buffer[-block_shift:] = sig1[idx * block_shift:(idx * block_shift) + block_shift]

    in2_buffer[:-block_shift] = in2_buffer[block_shift:]
    in2_buffer[-block_shift:] = sig2[idx * block_shift:(idx * block_shift) + block_shift]

    in3_buffer[:-block_shift] = in3_buffer[block_shift:]
    in3_buffer[-block_shift:] = sig3[idx * block_shift:(idx * block_shift) + block_shift]
    # calculate fft of input block

    in_block_fft1 = np.fft.rfft(in1_buffer)
    in_mag1 = np.abs(in_block_fft1)
    in_phase1 = np.angle(in_block_fft1)
    # reshape magnitude to input dimensions
    in_mag1 = np.reshape(in_mag1, (1, 1, -1)).astype('float32')

    in_block_fft2 = np.fft.rfft(in2_buffer)
    in_mag2 = np.abs(in_block_fft2)
    in_phase2 = np.angle(in_block_fft2)
    # reshape magnitude to input dimensions
    in_mag2 = np.reshape(in_mag2, (1, 1, -1)).astype('float32')

    in_block_fft3 = np.fft.rfft(in3_buffer)
    in_mag3 = np.abs(in_block_fft3)
    in_phase3 = np.angle(in_block_fft3)
    # reshape magnitude to input dimensions
    in_mag3 = np.reshape(in_mag3, (1, 1, -1)).astype('float32')

    # set tensors to the first model
    interpreter_1.set_tensor(input_details_1[1]['index'], states_1)
    interpreter_1.set_tensor(input_details_1[0]['index'], in_mag1)
    # run calculation
    interpreter_1.invoke()
    # get the output of the first block
    out_mask = interpreter_1.get_tensor(output_details_1[0]['index'])
    states_1 = interpreter_1.get_tensor(output_details_1[1]['index'])
    # calculate the ifft
    estimated_complex1 = in_mag1 * out_mask * np.exp(1j * in_phase1)
    estimated_complex2 = in_mag2 * out_mask * np.exp(1j * in_phase1)
    estimated_complex3 = in_mag3 * out_mask * np.exp(1j * in_phase1)
    estimated_block = np.fft.irfft(estimated_complex1)

    if bf==1:

        dump_speech = np.zeros((len(in1_buffer), len(CHANNEL_INDEX)))
        dump_speech[:, 0] = np.fft.irfft(estimated_complex1)
        dump_speech[:, 1] = np.fft.irfft(estimated_complex2)
        dump_speech[:, 2] = np.fft.irfft(estimated_complex3)

        out_mask = out_mask.reshape(257,)
        out_mask= tf.signal.frame(out_mask,FFT_LENGTH//2+1,FFT_SHIFT//2)

        complex_spectrum, _ = util.get_3dim_spectrum_from_data(dump_speech, FFT_LENGTH, FFT_SHIFT,
                                                          FFT_LENGTH)  # calcSpec(dump_speech.T)
        #print(out_mask.shape, complex_spectrum.shape)

        number_of_frame = np.shape(complex_spectrum)[1]
        synth = np.zeros(block_len).astype('float32')
        st = 0
        ed = FFT_LENGTH
        number_of_update = 0
        gg = []

        for i in range(0, number_of_frame):
            beamformer_maker.update_param(out_mask[i, :], np.expand_dims(complex_spectrum[:, i, :], 1))
            # number_of_update = number_of_update + 1
            # beamformer = beamformer_maker.get_mvdr_beamformer_by_higuchi()
            beamformer, c = beamformer_maker.get_mvdr_beamformer_by_higuchi_snr_selection()
            # beamformer[np.isnan(beamformer)] = 1.0
            # beamformer = beamformer / (np.abs(beamformer))
            # print('beamformer',beamformer)
            enhanced_speech = beamformer_maker.apply_beamformer(beamformer, complex_spectrum[:, i, :])
            # print('enhan',enhanced_speech)
            enhanced_speech[np.isnan(enhanced_speech)] = 0.0
            synth[st:ed] = synth[st:ed] + enhanced_speech
            st = st + FFT_SHIFT
            ed = ed + FFT_SHIFT

        estimated_block = synth

    out_buffer[:-block_shift] = out_buffer[block_shift:]
    out_buffer[-block_shift:] = np.zeros((block_shift))
    out_buffer += np.squeeze(estimated_block)
    # write block to output file
    ###if idx==0:
        ###print('1st block of out_buffer',out_buffer[:20])

    out_file[idx * block_shift:(idx * block_shift) + block_shift] = out_buffer[:block_shift]

    time_array.append(time.time() - start_time)
    #print((idx * block_shift))



if bf==1:
    sf.write(f'output/snr_(6)_MTR_TF_lite_f_b_f_bf.wav', out_file, fs)
else:
    sf.write(f'output/snr_(6)_MTR_TF_lite.wav', out_file, fs)
print('Processing Time [ms]:')
print(np.sum(np.stack(time_array)) * 1000)
print('Execution time per block: ' +
      str(np.round(np.mean(np.stack(time_array[10:])) * 1000, 2)) + ' ms')
print('Processing finished.')
