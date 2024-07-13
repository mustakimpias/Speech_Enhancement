# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import tensorflow as tf
from scipy.signal import stft, istft
from beamformer import util
import run_pesq_test as performance
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings("ignore")

SAMPLING_FREQUENCY = 16000
fs = 16000
FFT_LENGTH = 512
FFT_SHIFT = 128


# load model
model = tf.saved_model.load('./pretrained_model/dtln_saved_model')
infer = model.signatures["serving_default"]

#=========================================================================
#multi channel noisy signal read


#=============================================================================


block_len = 512
block_shift = 128

sig1 = sf.read('16k3micRecordingsFixedSpeechLevelALL/mainMicNoisyMTR6.wav',dtype='float32')[0]
sig2 = sf.read('16k3micRecordingsFixedSpeechLevelALL/inEarMicNoisyMTR6.wav',dtype='float32')[0]
sig3 = sf.read('16k3micRecordingsFixedSpeechLevelALL/noiseMicNoisyMTR6.wav',dtype='float32')[0]
clean = sf.read('16k3micRecordingsFixedSpeechLevelALL/mainMicSpeech.wav', dtype='float32')[0]

ll = min(len(sig1),len(sig2),len(sig3))
ll = block_shift*(ll//block_shift)
#print('ll',ll)
sig1=sig1[0:ll]
sig2=sig2[0:ll]
sig3=sig3[0:ll]
out_file = np.zeros(ll).astype('float32')

# create buffer

in1_buffer = np.zeros((block_len)).astype('float32')
in2_buffer = np.zeros((block_len)).astype('float32')
in3_buffer = np.zeros((block_len)).astype('float32')
out_buffer = np.zeros((block_len)).astype('float32')
# calculate number of blocks
num_blocks = ll // block_shift


time_array = []
#window = np.hanning(FFT_LENGTH)
num_frames = int(len(sig1) / block_len)

for idx in tqdm(range(num_blocks)):
    start_time = time.time()
    in1_buffer[:-block_shift] = in1_buffer[block_shift:]
    in1_buffer[-block_shift:] = sig1[idx * block_shift:(idx * block_shift) + block_shift]

    in2_buffer[:-block_shift] = in2_buffer[block_shift:]
    in2_buffer[-block_shift:] = sig2[idx * block_shift:(idx * block_shift) + block_shift]

    in3_buffer[:-block_shift] = in3_buffer[block_shift:]
    in3_buffer[-block_shift:] = sig3[idx * block_shift:(idx * block_shift) + block_shift]

    in_block = in1_buffer #(in1_buffer+in2_buffer+in3_buffer)/3.0
    in_block = np.expand_dims(in_block, axis=0).astype('float32')
    #print(in_block.shape)
    out_block = infer(tf.constant(in_block))['conv1d_1']
    out_block = np.array(out_block)
    # shift values and write to buffer
    out_buffer[:-block_shift] = out_buffer[block_shift:]
    out_buffer[-block_shift:] = np.zeros((block_shift))
    out_buffer += np.squeeze(out_block)
    # write block to output file
    #if idx>=9:
    out_file[(idx) * block_shift:((idx) * block_shift) + block_shift] = out_buffer[:block_shift]
    time_array.append(time.time()-start_time)

# write to .wav file
out_file[-block_len:]=out_buffer




#out_file = out_file/np.max(np.abs(out_file)) * 0.90
sf.write('output_SNR/snr_(6)_MTR_RT_DTLN.wav',out_file,fs)

#print('\n\n\t\t\t\t\t\t\t Performance')
print('-' * 100)
clean = sf.read('16k3micRecordingsFixedSpeechLevelALL/mainMicSpeech.wav', dtype='float32')[0]
'''print(np.max(out_file),np.max(clean))
print(np.min(out_file),np.min(clean))'''

enhanced = out_file
print('RT_ DTLN')
performance.score(out_file,clean,fs,'RT_DTLN_test1')

print('\n\n\t\t\t\t\t\t\tProcessing Time [ms]:')
print('-' * 100)
print(np.mean(np.stack(time_array)) * 1000)
print('Processing finished.')


