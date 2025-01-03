
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import streamlit as st
from Tasks.commonFunction import saveSignalToTxt

def generateSinWave(amplitude, frequency, phase, sampling_frequency):
  
    duration=1
    n_samples =int( sampling_frequency * duration)
    n = np.arange(n_samples)  
    sine_wave = amplitude * np.sin(2 * pi * frequency * n / sampling_frequency + phase)
    return n, sine_wave

def generateCosWave(amplitude, frequency, phase, sampling_frequency):
 
    duration=1
    n_samples =int( sampling_frequency * duration)
    n = np.arange(n_samples)  
    cosine_wave = amplitude * np.cos(2 * pi * frequency * n / sampling_frequency + phase)
    return n, cosine_wave


def displaySignal():
   
    st.title("Signal Generation")

   
    amplitude = st.number_input("Amplitude:", min_value=0.0, value=1.0, step=0.1)
    phase = st.number_input("Phase Shift (radians):", min_value=0.0, max_value=2*pi, format="%.14f")
    frequency = st.number_input("Signal Frequency (Hz):", min_value=0.1, value=1.0, step=1.0)
    sampling_frequency = st.number_input("Sampling Frequency (Hz):", min_value=0.1, value=100.0, step=1.0)
   
    if sampling_frequency < 2 * frequency:
        st.error(f"Sampling Frequency must be at least twice the Signal Frequency ({2 * frequency} Hz). Please increase the Sampling Frequency.")

    st.text("Choose Signal Type:\n")
    generate_sine = st.checkbox("Generate Sine Wave")
    generate_cosine = st.checkbox("Generate Cosine Wave")

   
    if st.button("Generate Signal"):
        amplitude = float(amplitude) 

    
        figs, axs = plt.subplots(figsize=(10, 5)) 
        figc, axc = plt.subplots(figsize=(10, 5)) 
        
        if generate_sine:
            n_s, signal_s = generateSinWave(amplitude, frequency, phase, sampling_frequency)
            axs.plot(n_s[:20], signal_s[:20])  
            axs.set_title("Sine Wave")
            axs.stem(n_s[:20], signal_s[:20])
            axs.set_xlabel('Sample Index (n)')
            axs.set_ylabel('Amplitude')
            axs.legend()  
            axs.grid(True)
            plt.tight_layout()
            st.pyplot(figs)
            sinSignalOutput="sinOutputFile.txt"
            saveSignalToTxt(sinSignalOutput,n_s,signal_s,amplitude, frequency, phase, sampling_frequency)
            SignalSamplesAreEqual('SinOutput.txt',n_s,signal_s)

        if generate_cosine:
            n_c, signal_c = generateCosWave(amplitude, frequency, phase, sampling_frequency)
            axc.plot(n_c[:20], signal_c[:20])  
            axc.set_title("Cosine Wave")
            axc.stem(n_c[:20], signal_c[:20])
            axc.set_xlabel('Sample Index (n)')
            axc.set_ylabel('Amplitude')
            axc.legend()  
            axc.grid(True)
            plt.tight_layout()
            st.pyplot(figc)
            cosSignalOutput="cosOutputFile.txt"
            saveSignalToTxt(cosSignalOutput,n_c,signal_c,amplitude, frequency, phase, sampling_frequency)
            SignalSamplesAreEqual('CosOutput.txt',n_c,signal_c)


def SignalSamplesAreEqual(file_name,indices,samples):
    expected_indices=[]
    expected_samples=[]
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L=line.strip()
            if len(L.split(' '))==2:
                L=line.split(' ')
                V1=int(L[0])
                V2=float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
                
    if len(expected_samples)!=len(samples):
        print("Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(expected_samples)):
        if abs(samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Test case failed, your signal have different values from the expected one") 
            return
    print("Test case passed successfully")





