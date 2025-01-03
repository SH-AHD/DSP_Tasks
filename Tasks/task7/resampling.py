import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.signal import  freqz
from Tasks.task7.FIR import design_fir_filter,plot_filter_response
from Tasks.task6.Convolution.convfunc import convolve_signals
from Tasks.task7.CompareSignal import Compare_Signals
import tkinter as tk
from tkinter import messagebox
import io



def convolve_signal(x, x_indices, h_indices, h):
   
    print(type(x_indices))
    print(x_indices)
    N = len(x) + len(h) - 1
    y = np.zeros(N)
    conv_indices=np.zeros(N)
   
    for n in range(N):
        for m in range(len(h)):
            if n - m >= 0 and n - m < len(x):
                  y[n] += x[n - m] * h[m]
                  
   
    conv_indices = list(range(min(x_indices) + min(h_indices),
                                    max(x_indices) + max(h_indices) + 1))

    
    st.write("conv_indices\n",conv_indices)

    return y,conv_indices


# Function to save signal to file
def save_signal_to_file(signal, file_name="resampled_signal.txt"):
    indices = np.arange(len(signal))
    
   
    with open(file_name, 'w') as f:
        for i, sample in zip(indices, signal):
            f.write(f"{i} {sample}\n")

    st.write("saved")        


def design_low_pass_filter(FS, FC, TransitionBand, StopBandAttenuation):

    
    filter_coefficients,lines,N,windowtype =design_fir_filter("Low pass", FC, FS, TransitionBand, StopBandAttenuation)
    
    st.header('Filter Design Results')
        
        # Coefficients plot
    plt.figure(figsize=(10, 4))
    plt.stem(lines, filter_coefficients)
    plt.title('FIR Filter Coefficients')
    plt.xlabel('Tap Index')
    plt.ylabel('Coefficient Value')
    plt.grid(True)
    st.pyplot(plt)
    plt.close()

        # Frequency response plot
    freq_response_plot = plot_filter_response(filter_coefficients, FS)
    st.pyplot(freq_response_plot)
    plt.close()

        # Display additional information
    st.subheader('Filter Specifications')
    st.write(f"**Filter Type:** {"Low pass"}")
    st.write(f"**Window Type:** {windowtype}")
    st.write(f"**Filter Length:** {N}")
    st.write(f"**Coefficients:** {filter_coefficients}")
    


    return filter_coefficients , lines


def remove_tr_zeros(signal):
    return np.trim_zeros(signal, trim='b')




def upsample(signal, indx, L):
    # Create new indices for the upsampled signal
    start = indx[0]
    end = indx[0] + (len(indx) - 1) * L
    new_indices = list(range(start, end + 1))
    
    upsampled_signal = []
    for i in range(len(signal)):
        upsampled_signal.append(signal[i])
        
        if i < len(signal) - 1:
            upsampled_signal.extend([0] * (L - 1))
    
    return  upsampled_signal,new_indices


def downsample(filtered_signal, M, original_indices):

    
    downsampled_signal = filtered_signal[::M]
    downsampled_indices=[]
    return  downsampled_signal,downsampled_indices[:len(downsampled_signal)]
        
       


def resample_signal(signal, indx, lines, M, L, filter_coefficients):
    
   
    if L > 0 and M == 0:
        
        upsampled_signal,upsampled_indices = upsample(signal, indx, L)
        filtered_signal,filtered_indices=convolve_signal(upsampled_signal, upsampled_indices, lines, filter_coefficients)
        return filtered_signal, filtered_indices
    
    # Downsampling case
    elif M > 0 and L == 0:
        filtered_signal, filtered_indices=convolve_signal(signal, indx, lines, filter_coefficients)
        downsampled_signal, downsampled_indices = downsample(filtered_signal, M, filtered_indices)
        
       
        return downsampled_signal, downsampled_indices
    
    # Upsampling and downsampling case
    

    elif M > 0 and L > 0:
        
        upsampled_signal,upsampled_indices = upsample(signal, indx, L)
        filtered_signal,filtered_indices=convolve_signal(upsampled_signal, upsampled_indices, lines, filter_coefficients)
        downsampled_signal, downsampled_indices = downsample(filtered_signal, M, filtered_indices)
        
        return downsampled_signal, downsampled_indices





def parse_data(file):
    
    lines = file.readlines()
    samples= []
    indices = []
    for line in lines[3:]:
        line = line.decode('utf-8').strip()
        if line:
            parts = line.split()
            if len(parts) == 2:
                try:
                    value = float(parts[1])
                    samples.append(value)
                    index=int(parts[0])
                    indices.append(index)
                except ValueError:
                    continue
    return samples , indices


   

def resampling_application():
    st.title("Resampling Application")

   
    uploaded_file = st.file_uploader("Upload your ECG signal file (txt format)", type="txt")
    if uploaded_file is not None:
        
        signal,indx=parse_data(uploaded_file)
        st.write ("indx",indx)
        st.write ("signal",signal)
        
        st.header("Filter Specifications")
        FS = st.number_input("Enter the sampling frequency (FS) in Hz:", min_value=100, value=8000)
        FC = [st.number_input("Enter the cutoff frequency (FC) in Hz:", min_value=1, value=1500)]
        TransitionBand = st.number_input("Enter the transition band width in Hz:", min_value=1, value=500)
        StopBandAttenuation = st.number_input("Enter the stopband attenuation (dB):", min_value=0, value=50)

        filter_coefficients,lines = design_low_pass_filter(
            FS=FS,
            FC=FC,
            TransitionBand=TransitionBand,
            StopBandAttenuation=StopBandAttenuation
        )

        
        M = st.number_input("Enter decimation factor (M):", min_value=0, step=1, value=0)
        L = st.number_input("Enter interpolation factor (L):", min_value=0, step=1, value=0)

        if M == 0 and L == 0:
            st.error("Both M and L cannot be zero. Please provide valid values.")
            return

        
        resampled_signal, indices = resample_signal(signal,indx,lines ,M, L, filter_coefficients)
        
       
        if L > 0 and M==0:
            Compare_Signals("Tasks/task7/Sampling test cases/Testcase 2/Sampling_Up.txt", indices, resampled_signal)
    
        
        elif M > 0 and L==0:
            Compare_Signals("Tasks/task7/Sampling test cases/Testcase 1/Sampling_Down.txt", indices, resampled_signal)

        elif M>0 and L>0:
        
            Compare_Signals("Tasks/task7/Sampling test cases/Testcase 3/Sampling_Up_Down.txt", indices, resampled_signal)


              