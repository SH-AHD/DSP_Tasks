import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from Tasks.task6.Convolution.ConvTest import ConvTest
#from Tasks.commonFunction import saveSignalToTxt


def convolve_signals(x, h):
    # The length of the output signal will be len(x) + len(h) - 1
    N = len(x) + len(h) - 1
    y = np.zeros(N)
    
   
    for n in range(N):
        for m in range(len(h)):
            if n - m >= 0 and n - m < len(x):
                y[n] += x[n - m] * h[m]
    
    return y




def plot_signals(signal_before, signal_after, indices_before, indices_after):
   
    # Plotting the signal before convolution
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(indices_before, signal_before, label="Before Convolution", color="blue")
    plt.title("Signal Before Convolution")
    plt.xlabel("Indices")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    # Plotting the signal after convolution
    plt.subplot(2, 1, 2)
    plt.plot(indices_after, signal_after, label="After Convolution", color="red")
    plt.title("Signal After Convolution")
    plt.xlabel("Indices")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    plt.tight_layout()
    st.pyplot(plt)

def plot_signals(signal_before1, indices_before1, signal_before2, indices_before2, signal_after, indices_after):
    """
    Plots two signals before and after convolution on the same graph for comparison.
    
    Args:
    signal_before1 (list): First signal before convolution.
    indices_before1 (list): Indices for the first signal before convolution.
    signal_before2 (list): Second signal before convolution.
    indices_before2 (list): Indices for the second signal before convolution.
    signal_after (list): Convolved signal after convolution.
    indices_after (list): Indices for the convolved signal.
    """
    plt.figure(figsize=(14, 12))

    # Plot the first signal before convolution
    plt.subplot(3, 1, 1)
    plt.plot(indices_before1, signal_before1, marker='o', color='b', label="Signal 2 Before Convolution")
    #plt.title("Signal 1 Before Convolution")
    plt.xlabel("Indices")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    # Plot the second signal before convolution
    plt.subplot(3, 1, 2)
    plt.plot(indices_before2, signal_before2, marker='o', color='r', label="Signal 2 Before Convolution")
    #plt.title("Signal 2 Before Convolution")
    plt.xlabel("Indices")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    # Plot the result after convolution
    plt.subplot(3, 1, 3)
    plt.plot(indices_after, signal_after, marker='o', color='g', label="Signal After Convolution")
    #plt.title("Signal After Convolution")
    plt.xlabel("Indices")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()

    # Display the plot in Streamlit
    st.pyplot(plt)

def saveSignalToTxt(filename, signal_values):
    with open(filename, 'w') as file:
        file.write("0\n")
        file.write("0\n")
        N1 = len(signal_values)
        file.write(f"{N1}\n")
        for index, val in enumerate(signal_values):
            file.write(f"{index} {val:.6f}\n")
    st.write(f"Signal saved to {filename}")

def read_input_file(file):
    """
    Reads an input file containing signal indices and samples.
    Returns:
    tuple: Indices and samples of the signal.
    """
    indices = []
    samples = []
    
    lines = file.readlines()
        
    num_samples = int(lines[2].strip())  # Reading the number of samples
    for line in lines[3:3 + num_samples]:
            index, sample = line.strip().split()
            indices.append(int(index))  # Convert the index to integer
            samples.append(float(sample))  # Convert the sample to float
    
    return indices, samples



def convdisplay():
   
    InputSamplesSignal1=None
    InputIndicesSignal1=None
    InputIndicesSignal2=None
    InputSamplesSignal2=None
    st.title(" Convolve Signals:")
    st.write("### Step 1: Upload the input signal file")
    
    
    input_file1 = st.file_uploader("Upload the first signal file", type=['txt'], key="input_file1")
    input_file2 = st.file_uploader("Upload the second signal file", type=['txt'], key="input_file2")

    if input_file1 and input_file2:
       
        InputIndicesSignal1,InputSamplesSignal1=read_input_file(input_file1)
        InputIndicesSignal2,InputSamplesSignal2=read_input_file(input_file2)

        output_signal = convolve_signals(InputSamplesSignal1, InputSamplesSignal2)
        #filename="convolve_signals.txt"
        #saveSignalToTxt(filename,output_signal)
        
        output_indices = list(range(min(InputIndicesSignal1) + min(InputIndicesSignal2),
                                    max(InputIndicesSignal1) + max(InputIndicesSignal2) + 1))

       
        st.title("Signal Convolution Before and After")
        plot_signals(InputSamplesSignal1, InputIndicesSignal1, InputSamplesSignal2, InputIndicesSignal2, output_signal, output_indices)

        ConvTest(output_indices, output_signal)
