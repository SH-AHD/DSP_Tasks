import numpy as np
import streamlit as st
import math
import matplotlib.pyplot as plt
from Tasks.task6.Convolution.convfunc import convolve_signals
from Tasks.task7.CompareSignal import Compare_Signals
from scipy.signal import  freqz

def normalize_frequency(frequencies, fs):
    """Normalize frequency (F1, F2, or FC) based on the sampling frequency."""
   
    return [f / fs for f in frequencies]
  

def hamming_window(N,n):
    return [0.54 + 0.46 * math.cos(2 * math.pi * n_i / N) for n_i in n]


def hanning_window(N,n):
    return [0.5 + (0.5* math.cos(2 * math.pi * n_i / N)) for n_i in n]

def rectangular_window(N):
    return [1] * (N + 1)

def blackman_window(N,n):

    return [0.42 + 0.5 * math.cos(2 * math.pi * n_i / (N-1)) + 0.08 * math.cos(4 * math.pi * n_i / (N-1)) for n_i in n]


def choose_window(stop_att):
    if stop_att <= 21:
        return "rectangular"
    elif stop_att <= 44:
        return "hanning"
    elif stop_att <= 53:
        return "hamming"
    elif stop_att <= 74:
        return "blackman"
    else:
        raise ValueError("Unsupported attenuation level.")



def design_fir_filter(filter_type, fc, fs, transition_band, stopband_attenuation):
   
 
    if filter_type == "Low pass":
        fc = [fc[0] + transition_band / 2]
    elif filter_type == "High pass":
        fc = [fc[0] - transition_band / 2]
    elif filter_type == "Band pass":
        fc = [fc[0] - transition_band / 2, fc[1] + transition_band / 2]
    elif filter_type == "Band stop":
        fc = [fc[0] + transition_band / 2, fc[1] - transition_band / 2]
    else:
        raise ValueError("Unknown filter type")

    fc = [f / fs for f in fc]  # Normalize frequencies

    
    trans_width = transition_band / fs
    if stopband_attenuation <= 21:
        N = int(np.ceil(0.9 / trans_width))
    elif stopband_attenuation <= 44:
        N = int(np.ceil(3.1 / trans_width))
    elif stopband_attenuation <= 53:
        N = int(np.ceil(3.3 / trans_width))
    elif stopband_attenuation <= 74:
        N = int(np.ceil(5.5 / trans_width))
    else:
        raise ValueError("Unsupported stop band attenuation.")
    if N % 2 == 0:
        N += 1

    
    lines = np.arange(-(N // 2), N // 2 + 1)
    
    coefficients = np.zeros_like(lines, dtype=float)

    if filter_type == "Low pass":
        coefficients[lines == 0] = 2 * fc[0]

        coefficients[lines != 0] = 2 * fc[0] * np.sin(2 * np.pi * fc[0] * lines[lines != 0]) / (2 * np.pi * fc[0] * lines[lines != 0])

    elif filter_type == "High pass":
        coefficients[lines == 0] = 1 - 2 * fc[0]

        coefficients[lines != 0] = -2 * fc[0] * np.sin(2 * np.pi * fc[0] * lines[lines != 0]) / (2 * np.pi * fc[0] * lines[lines != 0])

    elif filter_type == "Band pass":
        coefficients[lines == 0] = 2 * (fc[1] - fc[0])

        coefficients[lines != 0] = (
                2 * fc[1] * np.sin(2 * np.pi * fc[1] * lines[lines != 0]) / (2 * np.pi * fc[1] * lines[lines != 0])
                - 2 * fc[0] * np.sin(2 * np.pi * fc[0] * lines[lines != 0]) / (2 * np.pi * fc[0] * lines[lines != 0])
        )

    elif filter_type == "Band stop":
        coefficients[lines == 0] = 1 - 2 * (fc[1] - fc[0])

        coefficients[lines != 0] = (
                2 * fc[0] * np.sin(2 * np.pi * fc[0] * lines[lines != 0]) / (2 * np.pi * fc[0] * lines[lines != 0])
                - 2 * fc[1] * np.sin(2 * np.pi * fc[1] * lines[lines != 0]) / (2 * np.pi * fc[1] * lines[lines != 0])
        )

    else:
        raise ValueError("Unknown filter type")

    # Apply window
    window = choose_window(stopband_attenuation)
    if window == "rectangular":
        w=rectangular_window(N)
    elif window == "hanning":
        w=hanning_window(N,lines)
    elif window == "hamming":
        w=hamming_window(N,lines)
    elif window == "blackman":
        w=blackman_window(N,lines)

    if len(w) != len(coefficients):
        w = np.resize(w, coefficients.shape)

    coeff = coefficients * w


    return coeff,lines, N,window



def plot_filter_response(coeff, fs):
    w, h = freqz(coeff)
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.title('Magnitude Response')
    plt.plot(0.5 * fs * w / np.pi, np.abs(h))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.title('Phase Response')
    plt.plot(0.5 * fs * w / np.pi, np.angle(h))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (radians)')
    plt.grid(True)
    plt.tight_layout()
    return plt



def parse_filter_data(file):
    """Parse the input data (indices and coefficients) and return as a dictionary or arrays."""
    
   
    lines = file.readlines()
    
    
    
    indices = []
    coefficients = []

    
    for line in lines:
        parts = line.split()
        
        
        if len(parts) == 1:
            indices.append(int(parts[0]))
            coefficients.append(0.0)  
        elif len(parts) == 2:
            index = int(parts[0])
            coefficient = float(parts[1])
            indices.append(index)
            coefficients.append(coefficient)
    return np.array(indices), np.array(coefficients)



def save_filtered_signal(output_data,N, output_file):
    lines=[]
    with open(output_file, 'w') as f:
        f.write("0\n")
        f.write("0\n")
        
        
        for idx, coeff in zip(N, output_data):
            f.write(f"{idx} {coeff:.8f}\n")
       
def read_ecg_file(file_path):
    """Reads ECG signal data from the provided file."""
    try:
        data = np.loadtxt(file_path, delimiter=None)
        if data.ndim > 1:
            data = data[:, 0]
        return data.flatten()
    except ValueError as e:
        raise ValueError(f"Error reading file: {e}")


def fir_filter_ui():
    """Streamlit user interface for FIR filtering application."""
    st.title("FIR Filtering Application")

    test_type = st.radio("Select Test Type:", ["Test type 1: Specification Only", "Test type 2: Specification and Signal"])

    if test_type == "Test type 1: Specification Only":
        spec_file = st.file_uploader("Upload Specification File (txt format):", type=["txt"])
        
        if spec_file:
            spec_data = spec_file.read().decode("utf-8")
            st.write("Specification Content:")
            st.code(spec_data)
            
            filter_type = st.selectbox("Filter Type:", ["Low pass", "High pass", "Band pass", "Band stop"])
            fs = st.number_input("Sampling Frequency (Hz):", value=1000.0)
            stopband_attenuation = st.number_input("Stopband Attenuation (dB):", value=50.0)
            
            if filter_type in ["Band pass", "Band stop"]:
                cutoff1 = st.number_input("Lower Cutoff Frequency (f1):", value=100.0)
                cutoff2 = st.number_input("Upper Cutoff Frequency (f2):", value=200.0)
                fc = [cutoff1, cutoff2]
            else:
                fc = [st.number_input("Cutoff Frequency:", value=100.0)]
            
            transition_band = st.number_input("Transition Band (Hz):", value=50.0)
            
            if st.button("Apply Filter"):
                coefficients, lines, N,window_type =design_fir_filter(filter_type, fc, fs, transition_band, stopband_attenuation)
                
                st.write(f"Filter Coefficients (N={N}, Window: {window_type}):")
                st.write(coefficients)

                file_name = f"{filter_type}_fir_coefficients_ONLY.txt"
                save_filtered_signal(coefficients,lines, file_name)
                st.success(f"Filter coefficients saved to {file_name}.")

                with open(file_name, "rb") as f:
                    st.download_button("Download Filter Coefficients", f, file_name=file_name)
                

                # Compare Signals based on filter type and parameters
                if filter_type == "Low pass":
                    Compare_Signals("Tasks/task7/FIR test cases/Testcase 1/LPFCoefficients.txt", lines, coefficients)
                elif filter_type == "High pass":
                    Compare_Signals("Tasks/task7/FIR test cases/Testcase 3/HPFCoefficients.txt", lines, coefficients)
                elif filter_type == "Band pass":
                    Compare_Signals("Tasks/task7/FIR test cases/Testcase 5/BPFCoefficients.txt", lines, coefficients)
                elif filter_type == "Band stop":
                    Compare_Signals("Tasks/task7/FIR test cases/Testcase 7/BSFCoefficients.txt", lines, coefficients)

    elif test_type == "Test type 2: Specification and Signal":
        spec_file = st.file_uploader("Upload Specification File (txt format):", type=["txt"])
        signal_file = st.file_uploader("Upload Signal File (txt format):", type=["txt"])

        if spec_file and signal_file:
            spec_data = spec_file.read().decode("utf-8")
            st.write("Specification Content:")
            st.code(spec_data)

            filter_type = st.selectbox("Filter Type:", ["Low pass", "High pass", "Band pass", "Band stop"])
            fs = st.number_input("Sampling Frequency (Hz):", value=1000.0)
            stopband_attenuation = st.number_input("Stopband Attenuation (dB):", value=50.0)

            if filter_type in ["Band pass", "Band stop"]:
                cutoff1 = st.number_input("Lower Cutoff Frequency (f1):", value=100.0)
                cutoff2 = st.number_input("Upper Cutoff Frequency (f2):", value=200.0)
                fc = [cutoff1, cutoff2]
            else:
                fc = [st.number_input("Cutoff Frequency:", value=100.0)]
              
            transition_band = st.number_input("Transition Band (Hz):", value=50.0)
            
            if st.button("Apply Filter"): 
                coefficients, lines, N,window_type =design_fir_filter(filter_type, fc, fs, transition_band, stopband_attenuation)
                 

                signal_indices , signal_data = parse_filter_data(signal_file)
                st.write("Uploaded Signal:")
                st.write(signal_data[:10])  # Display first 10 samples of signal data

                conv_result=convolve_signals(signal_data, coefficients)
                filtered_signal =remove_trailing_zeros(conv_result)
                st.write("Filtered Signal:")
                st.write(filtered_signal[:10])  # Display first 10 filtered values

                output_file =f"{filter_type}_filtered_signal.txt"
                save_filtered_signal(filtered_signal,lines, output_file)
                st.success(f"Filtered signal saved to {output_file}.")

                with open(output_file, "rb") as f:
                    st.download_button("Download Filtered Signal", f, file_name=output_file)

              
                # Compare Signals based on filter type and parameters
                if filter_type == "Low pass":
                    Compare_Signals("Tasks/task7/FIR test cases/Testcase 2/ecg_low_pass_filtered.txt", lines, filtered_signal)
                elif filter_type == "High pass":
                    Compare_Signals("Tasks/task7/FIR test cases/Testcase 4/ecg_high_pass_filtered.txt", lines, filtered_signal)
                elif filter_type == "Band pass":
                    Compare_Signals("Tasks/task7/FIR test cases/Testcase 6/ecg_band_pass_filtered.txt", lines, filtered_signal)
                elif filter_type == "Band stop":
                    Compare_Signals("Tasks/task7/FIR test cases/Testcase 8/ecg_band_stop_filtered.txt",lines, filtered_signal)


def remove_trailing_zeros(conv_result):
 
   
    conv_result = np.array(conv_result)
    
    result_without_zeros = conv_result[np.nonzero(conv_result)[0]]
    
    return result_without_zeros



