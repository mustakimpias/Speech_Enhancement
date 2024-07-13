
from tensorflow.lite.python import interpreter as tflite
import soundfile as sf
import numpy as np
from tqdm import tqdm
from beamformer import complexGMM_mvdr_snr_selective as cgmm_snr
import matplotlib.pyplot as plt
import warnings

import time
warnings.filterwarnings("ignore")

##########################
# the values are fixed, if you need other values, you have to retrain.
# The sampling rate of 16k is also fix.
FFT_LENGTH = 512
FFT_SHIFT = 128
block_len = 512
block_shift = 128
CHANNEL_INDEX = [1,2,3]
fs=16000

# load models
interpreter_1 = tflite.Interpreter(model_path='64units_f_50ol_relu.tflite', experimental_preserve_all_tensors=True)
interpreter_1.allocate_tensors()
interpreter_2 = tflite.Interpreter(model_path='64units_f_50ol_relu.tflite', experimental_preserve_all_tensors=True)
interpreter_2.allocate_tensors()
interpreter_3 = tflite.Interpreter(model_path='64units_f_50ol_relu.tflite', experimental_preserve_all_tensors=True)
interpreter_3.allocate_tensors()

# Get input and output tensors.
input_details_1 = interpreter_1.get_input_details()
output_details_1 = interpreter_1.get_output_details()
input_details_2 = interpreter_2.get_input_details()
output_details_2 = interpreter_2.get_output_details()
input_details_3 = interpreter_3.get_input_details()
output_details_3 = interpreter_3.get_output_details()

# create states for the lstms
states_1 = np.zeros(input_details_1[0]['shape']).astype('float32')
states_2 = np.zeros(input_details_2[0]['shape']).astype('float32')
states_3 = np.zeros(input_details_3[0]['shape']).astype('float32')

# load audio file at 16k fs (please change)
# Load data
sig1 = sf.read('16k3micRecordingsFixedSpeechLevelALL/mainMicNoisyMTR6.wav', dtype='float32')[0]
sig2 = sf.read('16k3micRecordingsFixedSpeechLevelALL/inEarMicNoisyMTR6.wav', dtype='float32')[0]
sig3 = sf.read('16k3micRecordingsFixedSpeechLevelALL/noiseMicNoisyMTR6.wav', dtype='float32')[0]
clean = sf.read('16k3micRecordingsFixedSpeechLevelALL/mainMicSpeech.wav', dtype='float32')[0]
noisy = sig1 + sig2 + sig3
ll = min(len(sig1), len(sig2), len(sig3), len(clean), len(noisy))

print('Total Signal duration',ll//fs ,'s' )
print('How many seconds would you like to process in each iteration?')
ss = int(input())
sft = ss*(ll//(ll//fs))
ll = sft*(ll//sft)
sig1=sig1[0:ll]
sig2=sig2[0:ll]
sig3=sig3[0:ll]
clean = clean[0:ll]
audio = sig1 + sig2 + sig3
noisy = audio

# file = '1_1'
# audio,fs = sf.read(f'./test/input/tarek_new/16k/{file}.wav')
# audio = audio[:,0]
# print(audio.shape)
# check for sampling rate
if fs != 16000:
    raise ValueError('This model only supports 16k sampling rate.')

# create buffer


in1_buffer = np.zeros((block_len)).astype('float32')
in2_buffer = np.zeros((block_len)).astype('float32')
in3_buffer = np.zeros((block_len)).astype('float32')
out_buffer = np.zeros((block_len)).astype('float32')
out_file = np.zeros(sft).astype('float32')
output = np.zeros(ll).astype('float32')

time_array = []

for itr in tqdm(range(0, ll, sft)):
    start_time = time.time()

    noisy = audio[itr:itr+sft]
    speech1 = sig1[itr:itr+sft]
    speech2 = sig2[itr:itr+sft]
    speech3 = sig3[itr:itr + sft]
    cln = clean[itr:itr + sft]
    #print('sft',sft)
    num_blocks = sft // block_shift
    # print(sft,num_blocks)
    mask1 = []
    mask2 = []
    mask3 = []
    for idx in range(num_blocks):
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

        # set tensors to the mic1
        interpreter_1.set_tensor(input_details_1[0]['index'], states_1)
        interpreter_1.set_tensor(input_details_1[1]['index'], in_mag1)
        # run calculation
        interpreter_1.invoke()

        model_tensors = {
            t['name']: interpreter_1.get_tensor(t['index'])
            for t in interpreter_1.get_tensor_details()
        }
        forsen = np.sum(model_tensors['model_1/lstm_3/lstm_cell_3/MatMul'][0] * model_tensors['model_1/lstm_3/unstack'])

        # get the output of the mic1
        mask = interpreter_1.get_tensor(output_details_1[0]['index'])
        states_1 = interpreter_1.get_tensor(output_details_1[1]['index'])
        mask1.append(mask)


        # set tensors to the mic2
        interpreter_2.set_tensor(input_details_2[0]['index'], states_2)
        interpreter_2.set_tensor(input_details_2[1]['index'], in_mag2)
        # run calculation
        interpreter_2.invoke()

        model_tensors = {
            t['name']: interpreter_2.get_tensor(t['index'])
            for t in interpreter_2.get_tensor_details()
        }
        forsen = np.sum(model_tensors['model_1/lstm_3/lstm_cell_3/MatMul'][0] * model_tensors['model_1/lstm_3/unstack'])

        # get the output of the mic2
        mask = interpreter_2.get_tensor(output_details_2[0]['index'])
        states_2 = interpreter_2.get_tensor(output_details_2[1]['index'])
        mask2.append(mask)


        # set tensors to the mic3
        interpreter_3.set_tensor(input_details_3[0]['index'], states_3)
        interpreter_3.set_tensor(input_details_3[1]['index'], in_mag3)
        # run calculation
        interpreter_3.invoke()

        model_tensors = {
            t['name']: interpreter_3.get_tensor(t['index'])
            for t in interpreter_3.get_tensor_details()
        }
        forsen = np.sum(model_tensors['model_1/lstm_3/lstm_cell_3/MatMul'][0] * model_tensors['model_1/lstm_3/unstack'])

        # get the output of the mic3
        mask = interpreter_3.get_tensor(output_details_3[0]['index'])
        states_3 = interpreter_1.get_tensor(output_details_3[1]['index'])
        mask3.append(mask)

    mask1 = np.array(mask1)
    aa, _, _,bb = np.shape(np.array(mask1))
    mask1 = np.reshape(mask1,(aa,bb))
    mask2 = np.array(mask2)
    mask2 = np.reshape(mask2, (aa, bb))
    mask3 = np.array(mask3)
    mask3 = np.reshape(mask3, (aa, bb))

    sp_median = np.zeros((aa, bb, len(CHANNEL_INDEX)))
    sp_median[:, :, 0] = mask1
    sp_median[:, :, 1] = mask2
    sp_median[:, :, 2] = mask3
    sp_median_s = np.max(sp_median, axis=2)
    #print('mask',sp_median_s)
    n_median_s = 1 - sp_median_s

    '''figure, axis = plt.subplots(2, 1)

    img = axis[0].imshow(sp_median_s.T, origin="lower", aspect="auto")
    axis[0].set_title('MTR speech mask')

    img = axis[1].imshow(n_median_s.T, origin="lower", aspect="auto")
    axis[1].set_title('MTR noise mask')

    plt.colorbar(img, ax=axis)
    plt.show()'''

    # print('beamforming')
    dump_speech = np.zeros((len(speech1), len(CHANNEL_INDEX)))
    dump_speech[:, 0] = speech1
    dump_speech[:, 1] = speech2
    dump_speech[:, 2] = speech3

    cgmm_bf_snr = cgmm_snr.complexGMM_mvdr(fs, FFT_LENGTH, FFT_SHIFT)
    # print('dump speech',dump_speech.shape, dump_speech)
    # print('mask', sp_median_s.shape, sp_median_s)

    tmp_complex_spectrum, R_x, R_n, tt, nn = cgmm_bf_snr.get_spatial_correlation_matrix_from_mask_for_LSTM(
        dump_speech,
        speech_mask=sp_median_s.T,
        noise_mask=n_median_s.T,
        less_frame=0)

    #print('complex',tmp_complex_spectrum)
    selected_beamformer = cgmm_bf_snr.get_mvdr_beamformer_by_maxsnr(R_x, R_n)
    #print('Rx',R_x)
    enhan_speech2 = cgmm_bf_snr.apply_beamformer(selected_beamformer, tmp_complex_spectrum)
    enhan_speech2 = enhan_speech2 / np.max(np.abs(enhan_speech2)) * 0.65
    #print('speech',enhan_speech2)

    mv = np.size(enhan_speech2)
    output[itr:itr + mv] = enhan_speech2
    time_array.append(time.time() - start_time)

# write to .wav file
def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text

prefix = 'output/snr_(6)_MTR_TF_lite_bf_64units_f_50ol_relu_{}sec.wav'
d = { "{}": str(ss)}

prefix = replace_all(prefix, d)
#print(prefix)

sf.write(prefix, output, fs)


print('Processing Time [ms]:')
print(np.sum(np.stack(time_array)) * 1000)
'''print('Execution time per block: ' +
      str(np.round(np.mean(np.stack(time_array[10:])) * 1000, 2)) + ' ms')'''
print('Processing finished.')
