import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from Tasks.task6.Convolution.convfunc import convdisplay

def read_signal(file):
    data = []
    file_content = file.read().decode("utf-8")  
    for line in file_content.splitlines():
        values = line.strip().split(' ')
        if len(values) == 2:
            index = int(values[0])
            sample = float(values[1])
            data.append((index, sample))
    return zip(*data)


def compute_moving_average(signal, window_size):
    smoothed_signal = []
    for i in range(len(signal) - window_size + 1):
        window_average = np.mean(signal[i:i + window_size])  
        smoothed_signal.append(round(window_average))  
    return np.array(smoothed_signal)

def remove_dc_time(signal):
    mean_value = np.mean(signal)  
    return signal - mean_value 
import numpy as np

def normalized_cross_correlation(x1, x2):
    N = len(x1)
    r12 = np.zeros(N)
    
    
    for n in range(N):
        r12[n] = sum(x1[k] * x2[(k + n) % N] for k in range(N))
    
    
    norm_factor = np.sqrt(sum(x1[j]**2 for j in range(N)) * sum(x2[j]**2 for j in range(N)))
    
    pr12 = r12 / norm_factor
    
    return pr12

 

def save_signal_to_file(indices, samples, filename):
    with open(filename, "w") as f:
        for i, sample in zip(indices, samples):
            f.write(f"{i} {sample:.3f}\n")  # تنسيق القيم إلى 3 خانات عشرية

def compare_signals(real_file, computed_samples):
    real_indices = []
    real_samples = []

    file_content = real_file.read().decode("utf-8")
    for line in file_content.splitlines():
        values = line.strip().split()
        if len(values) == 2:
            real_indices.append(int(values[0]))
            real_samples.append(float(values[1]))

    
    if len(real_samples) != len(computed_samples):
        st.error("Test case failed: Length mismatch between real and computed signals.")
        return

    for i in range(len(real_samples)):
        if not np.isclose(real_samples[i], computed_samples[i], atol=0.01):
            st.error(f"Test case failed at index {real_indices[i]}: "
                     f"real {real_samples[i]} vs computed {computed_samples[i]}")
            return

    st.success("Test case passed successfully!")

def process_signal():
    operation = st.selectbox("Select operation", ("None","convolve", "Moving Average", "Cross-Correlation", "Remove DC Component (Time Domain)"))

    if operation=="convolve":
        convdisplay()
    else:
        uploaded_file = st.file_uploader("Upload your signal file", type=["txt"])
       
        if uploaded_file is not None:
            indices, samples = read_signal(uploaded_file)
            
            if operation == "Remove DC Component (Time Domain)":
                dc_removed_signal = remove_dc_time(samples)
                st.subheader("Original Signal vs Signal without DC Component")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(indices, samples, label="Original Signal")
                ax.plot(indices, dc_removed_signal, label="Signal without DC Component", linestyle='--')
                ax.legend()
                st.pyplot(fig)

                
                result_file_name = "dc_removed_signal.txt"
                save_signal_to_file(indices, dc_removed_signal, result_file_name)
                with open(result_file_name, "rb") as f:
                    st.download_button(label="Download DC Removed Signal", data=f, file_name=result_file_name)

            
                uploaded_real_file = st.file_uploader("Upload the real signal file for comparison", type=["txt"])
                if uploaded_real_file is not None:
                    compare_signals(uploaded_real_file, dc_removed_signal)

            elif operation == "Moving Average":
                window_size = st.number_input("Enter window size for Moving Average:", min_value=1, value=3)
                smoothed_samples = compute_moving_average(samples, window_size)
                st.subheader("Original Signal vs Smoothed Signal (Moving Average)")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(indices, samples, label="Original Signal")
                ax.plot(indices[:len(smoothed_samples)], smoothed_samples, label="Smoothed Signal", linestyle='--')
                ax.legend()
                st.pyplot(fig)

                
                result_file_name = "smoothed_signal.txt"
                save_signal_to_file(indices[:len(smoothed_samples)], smoothed_samples, result_file_name)
                with open(result_file_name, "rb") as f:
                    st.download_button(label="Download Smoothed Signal", data=f, file_name=result_file_name)

                uploaded_real_file = st.file_uploader("Upload the real signal file for comparison", type=["txt"])
                if uploaded_real_file is not None:
                    compare_signals(uploaded_real_file, smoothed_samples)

            elif operation == "Cross-Correlation":
                uploaded_file2 = st.file_uploader("Upload the second signal file for Cross-Correlation", type=["txt"])
                if uploaded_file2 is not None:
                    indices2, samples2 = read_signal(uploaded_file2)
                    correlation_result = normalized_cross_correlation(samples, samples2)
                    st.subheader("Cross-Correlation Result")
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(correlation_result)
                    ax.set_title("Cross-Correlation between Signals")
                    st.pyplot(fig)

                   
                    result_file_name = "cross_correlation.txt"
                    save_signal_to_file(range(len(correlation_result)), correlation_result, result_file_name)
                    with open(result_file_name, "rb") as f:
                        st.download_button(label="Download Cross-Correlation Result", data=f, file_name=result_file_name)
                    
                    
                    uploaded_real_file = st.file_uploader("Upload the real signal file for comparison", type=["txt"])
                    if uploaded_real_file is not None:
                        compare_signals(uploaded_real_file, correlation_result)

            
