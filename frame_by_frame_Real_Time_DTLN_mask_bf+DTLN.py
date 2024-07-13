import numpy as np
import soundfile as sf
from beamformer import Frame_by_frame_BF as fbf_mvdr
from beamformer import util
import run_pesq_test as performance
import time
import tensorflow as tf
from tqdm import tqdm
import warnings


warnings.filterwarnings("ignore")

##########################
# ==========================================
# ANALYSIS PARAMETERS
# ==========================================

#128*32=1.77,5.2
FFT_LENGTH = 128
FFT_SHIFT = 32
block_len = 512
block_shift = 128
CHANNEL_INDEX = [1,2,3]
fs=16000

# load model
model = tf.saved_model.load('./pretrained_model/dtln_saved_model')
infer = model.signatures["serving_default"]
beamformer_maker = fbf_mvdr.frame_by_frame_mvdr(fs,
                                                    FFT_LENGTH,
                                                    FFT_SHIFT,
                                                    len(CHANNEL_INDEX))
beamformer = np.ones((len(CHANNEL_INDEX), FFT_LENGTH // 2 + 1), dtype=np.complex64)


#=====================
# Load data
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




# check for sampling rate
if fs != 16000:
    raise ValueError('This model only supports 16k sampling rate.')
# preallocate output audio


# create buffer
in1_buffer = np.zeros((block_len)).astype('float32')
in2_buffer = np.zeros((block_len)).astype('float32')
in3_buffer = np.zeros((block_len)).astype('float32')
out_buffer = np.zeros((block_len)).astype('float32')

in_buffer1 = np.zeros((block_len)).astype('float32')
out_buffer1 = np.zeros((block_len)).astype('float32')

output = np.zeros(ll).astype('float32')




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


num_blocks = ll//block_shift
#print(num_blocks)
time_array = []
for idx in tqdm(range(num_blocks)):
    #print('idx',idx)
    start_time = time.time()

    # shift values and write to buffer
    in1_buffer[:-block_shift] = in1_buffer[block_shift:]
    in1_buffer[-block_shift:] = sig1[idx * block_shift:(idx * block_shift) + block_shift]

    in2_buffer[:-block_shift] = in2_buffer[block_shift:]
    in2_buffer[-block_shift:] = sig2[idx * block_shift:(idx * block_shift) + block_shift]

    in3_buffer[:-block_shift] = in3_buffer[block_shift:]
    in3_buffer[-block_shift:] = sig3[idx * block_shift:(idx * block_shift) + block_shift]
    in_block = in1_buffer #(in1_buffer+in2_buffer+in3_buffer)/3.0

    in_block = np.expand_dims(in_block, axis=0).astype('float32')

    #print('in bloc 1',in_block)
    out_block = infer(tf.constant(in_block))['conv1d_1']
    out_block = np.array(out_block)
    #print('out block 1', out_block)

    out_buffer[:-block_shift] = out_buffer[block_shift:]
    out_buffer[-block_shift:] = np.zeros((block_shift))
    out_buffer += np.squeeze(out_block)



    mask1, tmp = calcMask(in1_buffer, out_buffer)
    mask2, _ = calcMask(in2_buffer, out_buffer)
    mask3, _ = calcMask(in3_buffer, out_buffer)
    aa, bb = np.shape(mask1)
    sp_median = np.zeros((aa, bb, len(CHANNEL_INDEX)))
    sp_median[:, :, 0] = mask1
    sp_median[:, :, 1] = mask2
    sp_median[:, :, 2] = mask3
    sp_median_s = np.max(sp_median, axis=2)
    n_median_s = 1 - sp_median_s
    #print('mask',sp_median_s)


    #print('beamforming')
    dump_speech = np.zeros((len(in1_buffer), len(CHANNEL_INDEX)))
    dump_speech[:, 0] = in1_buffer
    dump_speech[:, 1] = in2_buffer
    dump_speech[:, 2] = in3_buffer



    complex_spectrum, _= util.get_3dim_spectrum_from_data(dump_speech,FFT_LENGTH,FFT_SHIFT,FFT_LENGTH) #calcSpec(dump_speech.T)
    #print(sp_median_s.shape, complex_spectrum.shape)

    number_of_frame = np.shape(complex_spectrum)[1]
    synth = np.zeros(block_len).astype('float32')
    st = 0
    ed = FFT_LENGTH
    number_of_update = 0
    gg = []

    for i in range(0, number_of_frame):
        beamformer_maker.update_param(sp_median_s[i, :], np.expand_dims(complex_spectrum[:, i, :], 1))
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


    in_buffer1[:-block_shift] = in_buffer1[block_shift:]
    in_buffer1[-block_shift:] = synth[FFT_LENGTH:block_shift+FFT_LENGTH]
    in_block1 = np.expand_dims(in_buffer1, axis=0).astype('float32')
    out_block1 = infer(tf.constant(in_block1))['conv1d_1']

    out_block1 = np.array(out_block1)
    #print('out block 2',out_block1)
    # shift values and write to buffer
    out_buffer1[:-block_shift] = out_buffer1[block_shift:]
    out_buffer1[-block_shift:] = np.zeros((block_shift))
    out_buffer1 += np.squeeze(out_block1)
    # write block to output file
    output[(idx) * block_shift:((idx) * block_shift) + block_shift] = out_buffer1[:block_shift]
    time_array.append(time.time() - start_time)


prefix = 'output/snr_(6)_MTR_f_b_f_bf_DTLN_mask.wav'
sf.write(prefix, output, fs)


print('\n\n\t\t\t\t\t\t\t Performance')
print('-' * 100)

title = 'Final_f_by_f_DNLN_mask_bf'
enhanced = output
performance.score(enhanced,clean,fs,title)

print('\n\n\t\t\t\t\t\t\tProcessing Time [ms]:')
print('-' * 100)
print(np.mean(np.stack(time_array)) * 1000)
print('Processing finished.')