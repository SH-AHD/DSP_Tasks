import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import Tasks.task4.signalcompare as sc
from Tasks.task5.Task.comparesignals import SignalSamplesAreEqual as scom
from Tasks.commonFunction import saveSignalToTxt


def SignalSamplesAreEqualStreamlit(expected_samples, result_samples):
    Amp = sc.SignalComapreAmplitude(SignalInput=[], SignalOutput=[])
    phase = sc.SignalComaprePhaseShift(SignalInput=[], SignalOutput=[])
    
    if Amp & phase:
        st.write("Test case passed successfully!")
        return "Passed"
    
    if Amp == True and phase == False:
        st.write("Test case failed: The signals have different PhaseShift.")
        return "Failed"
    
    if phase == True and Amp == False:
        st.write("Test case failed: The signals have different Amplitude.")
        return "Failed"

def read_file_space(file):
    lines = file.readlines()
    samples = []
    
    for line in lines[3:]:
        line = line.decode('utf-8').strip()
        if line:
            parts = line.split()
            if len(parts) == 2:
                try:
                    value = float(parts[1])
                    samples.append(value)
                except ValueError:
                    continue
    return samples

def saveSignalToTxt(filename, signal_values):
    with open(filename, 'w') as file:
        file.write("0\n")
        file.write("0\n")
        N1 = len(signal_values)
        file.write(f"{N1}\n")
        for index, val in enumerate(signal_values):
            file.write(f"{index} {val:.6f}\n")
    st.write(f"Signal saved to {filename}")

def read_file_comma(file):
    lines = file.readlines()
    samples = []

    lines = [line.decode('utf-8').strip() for line in lines]
    for line in lines[4:]:
        if line.strip():
            parts = line.split(',')
            if len(parts) == 2:
                try:
                    amplitude = float(parts[0].replace('f', ''))
                    phase = float(parts[1].replace('f', ''))
                    real = amplitude * np.cos(phase)
                    imag = amplitude * np.sin(phase)
                    complex_value = real + 1j * imag
                    samples.append(complex_value)
                except ValueError:
                    continue
    return samples

def compute_fourier(input_samples, N, inverse=False):
    fourier_result = []
    for k in range(N):
        real_sum = 0
        imag_sum = 0
        # for n in range(N):
        #     angle = (2 if inverse else -2) * np.pi * k * n / N
        #     real_sum += input_samples[n] * np.cos(angle)
        #    # imag_sum += input_samples[n] * np.sin(angle)
        #     imag_sum += (input_samples[n] * np.sin(angle) if inverse else -input_samples[n] * np.sin(angle))
        for n in range(N):
            angle = 2 * np.pi * k * n / N  # Always positive angle
            real_sum += input_samples[n] * np.cos(angle)
            imag_sum += (input_samples[n] * np.sin(angle) if inverse else -input_samples[n] * np.sin(angle))
            
        
        
        if inverse:
            real_sum /= N
            imag_sum /= N

        # Store the complex result
        
        X_k = real_sum + 1j * imag_sum
        fourier_result.append(X_k)
        
    return fourier_result

def calculate_amplitude_and_phase(fourier_result):
    amplitude = [np.abs(X_k) for X_k in fourier_result]
    phase = [np.arctan2(X_k.imag, X_k.real) for X_k in fourier_result]  # Use atan2 for accurate phase calculation
    return amplitude, phase

def plot_frequency_analysis(freqs, amplitude, phase):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Discrete plot and continuous line for Amplitude
    axs[0].stem(freqs, amplitude, basefmt=" ", label="Discrete")
    axs[0].plot(freqs, amplitude, linestyle='-', color='blue', label="Continuous")
    axs[0].set_title('Frequency vs Amplitude')
    axs[0].set_xlabel('Frequency (Hz)')
    axs[0].set_ylabel('Amplitude')
    axs[0].set_xlim(0, np.max(freqs))
    axs[0].legend()

    # Discrete plot and continuous line for Phase
    axs[1].stem(freqs, phase, basefmt=" ", label="Discrete")
    axs[1].plot(freqs, phase, linestyle='-', color='orange', label="Continuous")
    axs[1].set_title('Frequency vs Phase')
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Phase (radians)')
    axs[1].set_xlim(0, np.max(freqs))
    axs[1].legend()

    st.pyplot(fig)

def plot_time_domain_comparison(t, input_samples, reconstructed_signal):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, input_samples, label='Original Signal')
    ax.stem(t, input_samples, basefmt=" ", label="Original Signal Discrete")
    ax.stem(t, reconstructed_signal, basefmt=" ", label="Reconstructed Signal Discrete")
    ax.plot(t, reconstructed_signal, label='Reconstructed Signal', linestyle='dashed')
    ax.set_title('Original vs Reconstructed Signal')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid()
    st.pyplot(fig)





def read_input_file(file):
    """
    Reads the input file and extracts samples.
    Expects the format with sample values starting from the fourth line.
    """
    lines = file.readlines()
    num_samples = int(lines[2].strip())
    samples = []

    for line in lines[3:3 + num_samples]:
        _, amplitude = line.strip().split()
        samples.append(float(amplitude))

    return samples

def compute_dct_custom(x):
    N = len(x)
    dct_result = np.zeros(N)
    factor = np.sqrt(2 / N)

    for k in range(N):
        sum_value = 0
        for n in range(N):
            angle = (np.pi / (4 * N)) * (2 * n - 1) * (2 *k - 1)
            sum_value += x[n] * np.cos(angle)
        dct_result[k] = factor * sum_value

    return dct_result

def save_dct_coefficients(filename, dct_coefficients):
    """
    Saves the selected DCT coefficients to a file.
    """
    with open(filename, 'w') as file:
        file.write("0\n")
        file.write("1\n")
        file.write(f"{len(dct_coefficients)}\n")
        for i, val in enumerate(dct_coefficients):
            file.write(f"0 {val:.6f}\n")
    st.write(f"DCT coefficients saved to {filename}")



def remove_dc_frequency_domain(input_samples):
  
    N = len(input_samples)

   #DFT 
    fourier_result = compute_fourier(input_samples, N, inverse=False)

    # Remove the DC component (set the first Fourier coefficient to 0)
    fourier_result[0] = 0

    # IDFT
    signal_without_dc = compute_fourier(fourier_result, N, inverse=True)

    # Return only the real part of the reconstructed signal
    return [np.real(x) for x in signal_without_dc]


def plot_signals(original_signal, processed_signal):
    # Create a figure and axes
    fig, axs = plt.subplots(2, 1, figsize=(10, 5))

    # Plot the original signal
    axs[0].plot(original_signal, color='blue', label="Original Signal")
    axs[0].set_title("Original Signal")
    axs[0].set_xlabel("Sample Index")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid()
    axs[0].legend()

    # Plot the processed signal
    axs[1].plot(processed_signal, color='green', label="Signal After Removing DC")
    axs[1].set_title("Signal After Removing DC Component")
    axs[1].set_xlabel("Sample Index")
    axs[1].set_ylabel("Amplitude")
    axs[1].grid()
    axs[1].legend()

    # Adjust layout
    plt.tight_layout()
    st.pyplot(fig)

def finalpresent():
    st.title(" Frequency Domain:")
    st.write("### Step 1: Upload the input signal file")
    input_file = st.file_uploader("Upload the input signal file", type=['txt'])

    # تعيين قيمة افتراضية لمتغير input_samples
    input_samples = None

    if input_file is not None:
        st.warning(" Please select DFT, IDFT , DCT or Remove the DC component before proceeding.")  # رسالة التحذير
        
        select = st.selectbox("Select DFT, IDFT , DCT or Remove the DC component ", ["DCT", "DFT", "IDFT","Remove the DC component"])

        # تأكد من أن الاختيار ليس فارغًا
        if select == "DFT":
            input_samples = read_file_space(input_file)
        elif select == "IDFT":
            input_samples = read_file_comma(input_file)
        elif select == "DCT":
            input_samples = read_input_file(input_file)
        elif select =="Remove the DC component":
            input_samples = read_input_file(input_file)

        if input_samples is not None and (select == "DFT"or select == "IDFT"):
            N = len(input_samples)
            sampling_frequency = st.number_input("Enter the sampling frequency in Hz:", min_value=1.0, step=1.0, value=20.0)
            
            t = np.linspace(0, 1, N, endpoint=False)

            # DFT Computation
            dft = compute_fourier(input_samples, N, inverse=False)
            amplitude, phase = calculate_amplitude_and_phase(dft)
            freqs = [k * sampling_frequency / N for k in range(N)]

            st.write("## Frequency Domain Analysis")
            plot_frequency_analysis(freqs, amplitude, phase)

            # IDFT and Signal Reconstruction
            reconstructed_signal = compute_fourier(input_samples, N, inverse=True)

            st.write("## Time Domain Comparison")
            plot_time_domain_comparison(t, input_samples, reconstructed_signal)

            st.write("### Step 2: Upload the expected output file for comparison")
            expected_output_file = st.file_uploader("Upload the expected output file", type=['txt'], key="expected")

            if expected_output_file is not None:
                if select == "DFT":
                    expected_output_samples = read_file_space(expected_output_file)
                else:
                    expected_output_samples = read_file_comma(expected_output_file)

                comparison_result = SignalSamplesAreEqualStreamlit(expected_output_samples, reconstructed_signal)
                st.write("## Comparison with Expected Output")
                st.write(comparison_result)
        
        
        if input_samples is not None and (select == "DCT"):
            dct_result=compute_dct_custom(input_samples)
            st.write("### DCT Result")
            st.write(dct_result)

            st.write("### Step 2: Select the number of DCT coefficients to save")
            m = st.slider("Select the number of coefficients (m)", min_value=1, max_value=len(dct_result), value=len(dct_result))
            
            # Display selected DCT coefficients
            st.write(f"First {m} DCT coefficients:")
            st.write(dct_result[:m])

            # Save to file
            if st.button("Save DCT Coefficients to File"):
                filename = "dct_coefficients.txt"
                save_dct_coefficients(filename, dct_result[:m])

            st.write(" Upload the expected output file for comparison")
            expected_output_file = st.file_uploader("Upload the expected output file", type=['txt'], key="expected")

            if expected_output_file is not None:
               
                expected_output_samples = read_input_file(expected_output_file)
                indices = list(range(m))  # Indices of selected coefficients
                scom("Tasks/task4/DCT/DCT_output.txt", indices, dct_result[:m])


        if input_samples is not None and select =="Remove the DC component":
                dcres=remove_dc_frequency_domain(input_samples)
                filename="DC removed.txt"
                saveSignalToTxt(filename,dcres)
                plot_signals(input_samples, dcres)
                indices = list(range(len(dcres)))
                scom("Tasks/task6/Remove DC component/DC_component_output.txt",indices,dcres)






