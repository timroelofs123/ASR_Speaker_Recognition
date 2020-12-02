import numpy as np
import soundfile as sf
from scipy.fftpack import fft, rfft, fftfreq
import librosa
import pickle
import glob, os


def windowing(data, frame_length, hop_size, windowing_function):
    """
    Method for spliting the speech into short segments and apply windowing functions on them.

    data: speech segment
    frame_length: length of the segments used for splitting the speech
    hop_size: overlap part of the segments
    windowing_function: Choose what type of windowing you apply to the segment
    """
    data = np.array(data)
    number_of_frames = 1 + np.int(
        np.floor((len(data) - frame_length) / hop_size)
    )  #Calculate the number of frames using the presented formula
    frame_matrix = np.zeros((frame_length, number_of_frames))

    if windowing_function == 'rect':
        window = np.ones((frame_length, 1))
    elif windowing_function == 'hann':
        window = np.hanning(frame_length)
    elif windowing_function == 'cosine':
        window = np.sqrt(np.hanning(frame_length))
    elif windowing_function == 'hamming':
        window = np.hamming(frame_length)

    for i in range(number_of_frames):
        start = i * hop_size
        stop = np.minimum(start + frame_length, len(data))

        frame = np.zeros(frame_length)

        frame[0:stop - start] = data[start:stop]
        frame_matrix[:, i] = np.multiply(window, frame)
    return frame_matrix


def spectrogram_func(frame_matrix):
    # Compute Spectrogram
    _, height = frame_matrix.shape
    for i in range(0, height):
        fourier_signal = fft(frame_matrix[:, i])
        magnitude = np.abs(fourier_signal)
        freq = 20 * np.log10(magnitude)
        frame_matrix[:, i] = freq
    return frame_matrix


def cepstrum_func(frame_matrix):
    eps = 0.00001  #Add this to the power spectrum to ensure values are above zero for log function

    _, height = frame_matrix.shape
    for i in range(0, height):
        #Compute real cepstrum of frame
        frame_matrix[:, i] = np.real(
            np.fft.ifft(
                np.log10(
                    np.absolute(
                        np.power(np.fft.fft(frame_matrix[:, i]), 2) + eps))))
    return frame_matrix


def energy_func(frame_matrix):
    _, height = frame_matrix.shape
    energy_list = []
    for i in range(0, height):
        frame = np.array(frame_matrix[:, i])
        frame = frame - np.mean(
            frame)  #Remove mean to omit effect of DC component

        sum_of_elements = 0
        for j in range(0, len(frame)):
            sum_of_elements += frame[j] * frame[j]

        energy_value = (np.sqrt(sum_of_elements)) / len(frame)  #Complete
        energy_list.append(energy_value)
    energy_list = np.array(energy_list)
    return energy_list


def mfcc_func(data, Fs):
    mfcc_data = librosa.feature.mfcc(data, Fs)
    return mfcc_data


def fundf_cepstrum_func(frame, fs, f0_min=80, f0_max=180, vuv_threshold=0):
    frame = np.array(frame)

    #Number of autocorrelation lag samples corresponding to f0_min
    max_lag = np.int(np.ceil(fs / f0_min))
    #Number of autocorrelation lag samples corresponding to f0_max
    min_lag = np.int(np.floor(fs / f0_max))

    eps = 0.00001  #Add this to the power spectrum to ensure values are above zero for log function

    #Compute real cepstrum of frame
    c = np.real(
        np.fft.ifft(np.log10(np.absolute(np.power(np.fft.fft(frame), 2) +
                                         eps))))

    #Locate cepstral peak and its amplitude between min_lag and max_lag
    c = c[min_lag:max_lag]

    cepstral_peak_val = np.amax(np.absolute(c))
    ind = np.argmax(c)

    if cepstral_peak_val > vuv_threshold:
        f0 = fs / (min_lag + ind)
    else:
        f0 = 0

    return f0, cepstral_peak_val


def fundamental_peaks_func(frame_matrix, Fs):
    _, height = frame_matrix.shape
    peaks = []
    for i in range(0, height):
        frame = np.array(frame_matrix[:, i])

        _, peak_val = fundf_cepstrum_func(frame, Fs)

        peaks.append(peak_val)
    peaks = np.array(peaks)
    return peaks


def main():
    """
    Function which goes through all the recordings and extracts the following:
    
    1. Spectrogram
    2. Cepstrum 
    3. Energy
    4. MFCC
    5. Fundamental Frequency

    All these features are saved in pickle files so that they will be easily accessed further.
    """
    for filename in glob.iglob('./LibriSpeech/dev-clean/**', recursive=True):
        if os.path.isfile(filename) and '.flac' in filename:  # filter dirs
            path = filename
            data, Fs = sf.read(path)
            eps = 0.00001  #Add this to the power spectrum to ensure values are above zero for log function

            # Windowing the Speech
            frame_length = np.int(np.around(0.025 * Fs))  # 25ms in samples
            hop_size = np.int(np.around(
                0.0125 * Fs))  # 12.5 ms in samples (50% overlap)
            window_types = ('rect', 'hann', 'cosine', 'hamming')
            frame_matrix = windowing(data, frame_length, hop_size,
                                     window_types[1])  # Windowing
            frame_matrix_windowing = frame_matrix.copy()

            # Compute Spectrogram
            width, height = frame_matrix.shape
            for i in range(0, height):
                fourier_signal = fft(frame_matrix[:, i])
                magnitude = np.abs(fourier_signal)
                freq = 20 * np.log10(magnitude + eps)
                frame_matrix[:, i] = freq

            # Compute Cepstrum
            cepstrum = frame_matrix.copy()

            width, height = cepstrum.shape
            for i in range(0, height):
                #Compute real cepstrum of frame
                cepstrum[:, i] = np.real(
                    np.fft.ifft(
                        np.log10(
                            np.absolute(
                                np.power(np.fft.fft(cepstrum[:, i]), 2) +
                                eps))))

            # Compute Energy of each frame
            width, height = frame_matrix_windowing.shape
            energy = []
            for i in range(0, height):
                frame = np.array(frame_matrix_windowing[:, i])
                frame = frame - np.mean(
                    frame)  #Remove mean to omit effect of DC component

                sum_of_elements = 0
                for j in range(0, len(frame)):
                    sum_of_elements += frame[j] * frame[j]

                energy_value = (np.sqrt(sum_of_elements)) / len(
                    frame)  #Complete
                energy.append(energy_value)
            energy = np.array(energy)

            # Compute MFCC
            mfcc_data = librosa.feature.mfcc(data, Fs)

            # Compute Fundamental Frequency Estimation
            peaks = []
            for i in range(0, height):
                frame = np.array(frame_matrix_windowing[:, i])

                _, peak_val = fundf_cepstrum_func(frame, Fs, 80, 180, 0)

                peaks.append(peak_val)
            peaks = np.array(peaks)

            # Save values obtained in a Pickle File

            new_name = filename.replace(".flac", ".pkl")

            with open(new_name, 'wb') as f:
                pickle.dump([frame_matrix, cepstrum, mfcc_data, energy, peaks],
                            f)


main()