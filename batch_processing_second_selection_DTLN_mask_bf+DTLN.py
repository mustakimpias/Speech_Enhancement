
import matplotlib.pyplot as pl
from beamformer import complexGMM_mvdr_snr_selective as cgmm_snr
import soundfile as sf
import numpy as np
import time
import tensorflow as tf
from tqdm import tqdm
import warnings

import run_pesq_test as performance

from pesq import pesq
from pystoi import stoi
import torch
import matplotlib.pyplot as plt
import torchaudio


warnings.filterwarnings("ignore")

##########################

SAMPLING_FREQUENCY = 16000
FFT_LENGTH = 512
FFT_SHIFT = 128
# the values are fixed, if you need other values, you have to retrain.
# The sampling rate of 16k is also fix.
block_len = 512
block_shift = 128
fs=16000



# ==========================================
# ANALYSIS PARAMETERS
# ==========================================
CHANNEL_INDEX = [1,2,3]
FFTL = FFT_LENGTH
SHIFT = FFT_SHIFT



def plot_spectrogram(sig, title="Spectrogram", xlim=None,i=0):
    sig=torch.from_numpy(sig)
    sig = sig.to(torch.double)
    N_FFT = FFT_LENGTH
    N_HOP = FFT_SHIFT
    stft = torchaudio.transforms.Spectrogram(
        n_fft=N_FFT,
        hop_length=N_HOP,
        power=None,
    )
    stft = stft(sig)
    magnitude = stft.abs()
    spectrogram = 20*torch.log10(magnitude + 1e-8).numpy()
    return spectrogram


def score(deg,title, ref,title1):
    #rate, deg = wavfile.read(enhanced)
    #rate, ref = wavfile.read(clean)
    #print('fs',rate)

    #print(ref.shape,deg.shape)
    ll = min(ref.shape[0],deg.shape[0])
    #print(ll)
    ref=ref[0:ll]
    deg=deg[0:ll]

    ref=np.reshape(ref,(-1,1)).T
    deg=np.reshape(deg,(-1,1)).T

    figure, axis = plt.subplots(2, 1)
    spectrogram =plot_spectrogram(ref)
    img = axis[0].imshow(spectrogram[0, :, :], origin="lower", aspect="auto")
    axis[0].set_title(title1)

    spectrogram = plot_spectrogram(deg)
    img = axis[1].imshow(spectrogram[0, :, :], origin="lower", aspect="auto")
    axis[1].set_title(title)

    plt.colorbar(img, ax=axis)
    plt.show()

#=====================

# load model
model = tf.saved_model.load('./pretrained_model/dtln_saved_model')
infer = model.signatures["serving_default"]

# Load data
sig1 = sf.read('16k3micRecordingsFixedSpeechLevelALL/mainMicNoisyMTR6.wav',dtype='float32')[0]
sig2 = sf.read('16k3micRecordingsFixedSpeechLevelALL/inEarMicNoisyMTR6.wav',dtype='float32')[0]
sig3 = sf.read('16k3micRecordingsFixedSpeechLevelALL/noiseMicNoisyMTR6.wav',dtype='float32')[0]
clean = sf.read('16k3micRecordingsFixedSpeechLevelALL/mainMicSpeech.wav', dtype='float32')[0]

ll = min(len(sig1),len(sig2),len(sig3),len(clean))

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
sf.write('output/noisy.wav',noisy/3.0,fs)
#print('ll sft',ll, sft)

# check for sampling rate
if fs != 16000:
    raise ValueError('This model only supports 16k sampling rate.')
# preallocate output audio


# create buffer
in_buffer = np.zeros((block_len)).astype('float32')
out_buffer = np.zeros((block_len)).astype('float32')

in_buffer1 = np.zeros((block_len)).astype('float32')
out_buffer1 = np.zeros((block_len)).astype('float32')

in_buffer2 = np.zeros((block_len)).astype('float32')
out_buffer2 = np.zeros((block_len)).astype('float32')

in_buffer3 = np.zeros((block_len)).astype('float32')
out_buffer3 = np.zeros((block_len)).astype('float32')



def calcMask(array1, array2):
    tmp = np.zeros((len(array1)))
    MASK = []
    bl = FFT_LENGTH
    bs = FFT_SHIFT
    nb = (array1.shape[0] - (bl - bs)) // bs
    for i in range(nb):
        fft1 = np.fft.rfft(array1[i * bs:(i * bs) + bl])
        inp = array1[i * bs:(i * bs) + bl]
        fft2 = np.fft.rfft(array2[i * bs:(i * bs) + bl])
        out = array2[i * bs:(i * bs) + bl]
        mag1 = np.abs(fft1)**2
        mag2 = np.abs(fft2)**2
        ang1 = np.angle(fft1)
        ang2 = np.angle(fft2)
        magMask = (mag2) / (mag2 + mag1)
        mask = magMask * np.exp(1j * ang2)
        tmp[i * bs:(i * bs) + bl] = np.fft.irfft(mask)
        MASK.append(magMask)
        #MASK.append(np.abs(out) / (np.abs(inp) + np.abs(out) + 1e-9))

    return np.array(MASK), tmp






#print('sft',sft)
out_file = np.zeros(sft).astype('float32')
out_file1 = np.zeros(sft).astype('float32')
out_file2 = np.zeros(sft).astype('float32')
out_file3 = np.zeros(sft).astype('float32')
output = np.zeros(ll).astype('float32')
# print('len',ll)
time_array = []
for itr in tqdm(range(0, ll, sft)):
    #print('ll sft itr, itr+sft', ll, sft, itr, itr+sft)
    start_time1 = time.time()
    noisy = audio[itr:itr+sft]
    speech1 = sig1[itr:itr+sft]
    speech2 = sig2[itr:itr+sft]
    speech3 = sig3[itr:itr + sft]
    cln = clean[itr:itr + sft]
    #print('sft',sft)
    num_blocks = sft // block_shift
    # print(sft,num_blocks)
    for idx in range(num_blocks):
        start_time = time.time()
        # shift values and write to buffer
        in_buffer[:-block_shift] = in_buffer[block_shift:]
        in_buffer[-block_shift:] = speech1[idx * block_shift:(idx * block_shift) + block_shift]
        in_block = np.expand_dims(in_buffer, axis=0).astype('float32')

        out_block = infer(tf.constant(in_block))['conv1d_1']
        out_block = np.array(out_block)
        out_buffer[:-block_shift] = out_buffer[block_shift:]
        out_buffer[-block_shift:] = np.zeros((block_shift))
        out_buffer += np.squeeze(out_block)
        out_file[(idx) * block_shift:((idx) * block_shift) + block_shift] = out_buffer[:block_shift]
        time_array.append(time.time() - start_time)



    output_dtln = out_file
    mask1, tmp = calcMask(speech1, out_file)
    mask2, _ = calcMask(speech2, out_file)
    mask3, _ = calcMask(speech3, out_file)
    aa, bb = np.shape(mask1)
    sp_median = np.zeros((aa, bb, len(CHANNEL_INDEX)))
    sp_median[:, :, 0] = mask1
    sp_median[:, :, 1] = mask2
    sp_median[:, :, 2] = mask3
    sp_median_s = np.max(sp_median, axis=2)
    n_median_s = 1 - sp_median_s

    '''figure, axis = plt.subplots(2, 1)

    img = axis[0].imshow(sp_median_s.T, origin="lower", aspect="auto")
    axis[0].set_title('MTR speech mask')

    img = axis[1].imshow(n_median_s.T, origin="lower", aspect="auto")
    axis[1].set_title('MTR noise mask')

    plt.colorbar(img, ax=axis)
    plt.show()'''


    #print('beamforming')
    dump_speech = np.zeros((len(speech1), len(CHANNEL_INDEX)))
    dump_speech[:, 0] = speech1
    dump_speech[:, 1] = speech2
    dump_speech[:, 2] = speech3

    cgmm_bf_snr = cgmm_snr.complexGMM_mvdr(SAMPLING_FREQUENCY, FFTL, SHIFT)
    # print('dump speech',dump_speech.shape, dump_speech)
    #print('mask', sp_median_s.shape, sp_median_s)


    tmp_complex_spectrum, R_x, R_n, tt, nn = cgmm_bf_snr.get_spatial_correlation_matrix_from_mask_for_LSTM(
        dump_speech,
        speech_mask=sp_median_s.T,
        noise_mask=n_median_s.T,
        less_frame=0)

    # print('complex',tmp_complex_spectrum.shape)
    selected_beamformer = cgmm_bf_snr.get_mvdr_beamformer_by_maxsnr(R_x, R_n)
    enhan_speech2 = cgmm_bf_snr.apply_beamformer(selected_beamformer, tmp_complex_spectrum)
    #enhan_speech2 = enhan_speech2 / np.max(np.abs(enhan_speech2)) * 0.65
    # print('idx'
    #print('enhan',enhan_speech2.shape)

    for idx in range(num_blocks-3):
        # shift values and write to buffer
        start_time = time.time()
        in_buffer[:-block_shift] = in_buffer[block_shift:]
        in_buffer[-block_shift:] = enhan_speech2[idx * block_shift:(idx * block_shift) + block_shift]
        in_block = np.expand_dims(in_buffer, axis=0).astype('float32')
        out_block = infer(tf.constant(in_block))['conv1d_1']

        out_block = np.array(out_block)
        # shift values and write to buffer
        out_buffer[:-block_shift] = out_buffer[block_shift:]
        out_buffer[-block_shift:] = np.zeros((block_shift))
        out_buffer += np.squeeze(out_block)
        # write block to output file
        out_file[(idx) * block_shift:((idx) * block_shift) + block_shift] = out_buffer[:block_shift]
        time_array.append(time.time() - start_time)
    output[itr:itr+sft] = out_file

    '''score(cln, 'Clean speech', noisy, 'MTR Noisy speech')
    score(output, 'MTR noisy MVDR BF DTLN enhanced speech', output_dtln, 'MTR noisy RT DTLN enhanced speech')'''

    time_array.append(time.time()-start_time1)





#prefix = 'output/TD_frame_by_frame_{}sec.wav'
def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text

prefix = 'output/snr_(6)_MTR_DTLN_mask_BP_FD_BF+DTLN_{}sec.wav'
d = { "{}": str(ss)}

prefix = replace_all(prefix, d)
#print(prefix)

sf.write(prefix, output, fs)


print('\n\n\t\t\t\t\t\t\t Performance')
print('-' * 100)




print('\n\n\t\t\t\t\t\t\tProcessing Time [ms]:')
print('-' * 100)
print(np.mean(np.stack(time_array)) * 1000)
print('Processing finished.')
