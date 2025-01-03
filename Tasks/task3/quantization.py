# quantization_functions.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Quantization Functions
def read_input_file(file):
    lines = file.readlines()
    samples = [float(line.split()[1]) for line in lines[3:] if line.strip()]
    return samples

# Test functions
def QuantizationTest1(file_contents, Your_EncodedValues, Your_QuantizedValues):
    expectedEncodedValues = []
    expectedQuantizedValues = []
    lines = file_contents.strip().split('\n')[3:]  # Skip the first three lines
    for line in lines:
        L = line.strip()
        if len(L.split(' ')) == 2:
            L = line.split(' ')
            expectedEncodedValues.append(L[0])
            expectedQuantizedValues.append(float(L[1]))

    if (len(Your_EncodedValues) != len(expectedEncodedValues) or
        len(Your_QuantizedValues) != len(expectedQuantizedValues)):
        return "QuantizationTest1 Test case failed, your signal has different length from the expected one"
    for i in range(len(Your_EncodedValues)):
        if Your_EncodedValues[i] != expectedEncodedValues[i]:
            return "QuantizationTest1 Test case failed, your EncodedValues differ from expected"
    for i in range(len(expectedQuantizedValues)):
        if abs(Your_QuantizedValues[i] - expectedQuantizedValues[i]) >= 0.01:
            return "QuantizationTest1 Test case failed, your QuantizedValues differ from expected"
    return "QuantizationTest1 Test case passed successfully"

def QuantizationTest2(file_contents, Your_IntervalIndices, Your_EncodedValues, Your_QuantizedValues, Your_SampledError):
    expectedIntervalIndices = []
    expectedEncodedValues = []
    expectedQuantizedValues = []
    expectedSampledError = []
    lines = file_contents.strip().split('\n')[3:]  # Skip the first three lines
    for line in lines:
        L = line.strip()
        if len(L.split(' ')) == 4:
            L = line.split(' ')
            expectedIntervalIndices.append(int(L[0]))
            expectedEncodedValues.append(L[1])
            expectedQuantizedValues.append(float(L[2]))
            expectedSampledError.append(float(L[3]))

    if (len(Your_IntervalIndices) != len(expectedIntervalIndices) or
        len(Your_EncodedValues) != len(expectedEncodedValues) or
        len(Your_QuantizedValues) != len(expectedQuantizedValues) or
        len(Your_SampledError) != len(expectedSampledError)):
        return "QuantizationTest2 Test case failed, your signal has different length from the expected one"
    for i in range(len(Your_IntervalIndices)):
        if Your_IntervalIndices[i] != expectedIntervalIndices[i]:
            return "QuantizationTest2 Test case failed, your indices differ from expected"
    for i in range(len(Your_EncodedValues)):
        if Your_EncodedValues[i] != expectedEncodedValues[i]:
            return "QuantizationTest2 Test case failed, your EncodedValues differ from expected"
    for i in range(len(expectedQuantizedValues)):
        if abs(Your_QuantizedValues[i] - expectedQuantizedValues[i]) >= 0.01:
            return "QuantizationTest2 Test case failed, your QuantizedValues differ from expected"
    for i in range(len(expectedSampledError)):
        if abs(Your_SampledError[i] - expectedSampledError[i]) >= 0.01:
            return "QuantizationTest2 Test case failed, your SampledError differ from expected"
    return "QuantizationTest2 Test case passed successfully"



def quantizationSignals(signal, num_levels, num_bits):
    min_val, max_val = min(signal), max(signal)
    interval_size = (max_val - min_val) / num_levels  #delta
    quantized_signal, encoded, errors, interval_indices = [], [], [], []

    for value in signal:
        interval_index = int((value - min_val) / interval_size) 
        interval_index = min(max(interval_index, 0), num_levels - 1)
        quantized_value = min_val + (interval_index + 0.5) * interval_size
        binary_code = format(interval_index, f'0{num_bits}b')
        error = quantized_value - value

        quantized_signal.append(round(quantized_value, 3))
        encoded.append(binary_code)
        errors.append(round(error, 3))
        interval_indices.append(interval_index + 1)

    return quantized_signal, encoded, errors, interval_indices

def quantizationApp():
    st.title("Quantization Application with Plot and Tests")

    # Upload the input file
    input_file = st.file_uploader("Upload the input file", type=["txt"])

    if input_file is not None:
        samples = read_input_file(input_file)

        # Choose quantization method
        choice = st.radio("Quantization Input", ("Number of Levels", "Number of Bits"))

        if choice == "Number of Levels":
            levels = st.number_input("Enter the number of levels:", min_value=2, step=1)
            num_bits = int(np.ceil(np.log2(levels)))
        else:
            num_bits = st.number_input("Enter the number of bits:", min_value=1, step=1)
            levels = 2 ** num_bits

        if levels:
            quantizedSamples, encoded, quantizationError, intervalIndexForSamples = quantizationSignals(samples, levels, num_bits)

            # Display results
            st.subheader("Quantization Results")
            quantization_df = pd.DataFrame({
                "Interval Index": intervalIndexForSamples,
                "Encoded Value": encoded,
                "Quantized Value": quantizedSamples,
                "Quantization Error": quantizationError
            })
            st.table(quantization_df)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(samples, label="Original Signal", marker='o', linestyle='-', color='blue')
            ax.plot(quantizedSamples, label="Quantized Signal", marker='x', linestyle='--', color='red')
            ax.set_xlabel("Sample Index")
            ax.set_ylabel("Signal Amplitude")
            ax.set_title("Original vs. Quantized Signal")
            ax.legend()
            st.pyplot(fig)

            output_text = "\n".join([f"{enc} {quant:.2f}" for enc, quant in zip(encoded, quantizedSamples)])
            st.download_button("Download Quantized Output", data=output_text, file_name="output.txt")

            test_file = st.file_uploader("Upload expected output file for testing", type=["txt"])

            if test_file:
                test_file_contents = test_file.getvalue().decode("utf-8")
                if choice == "Number of Levels":
                   result = QuantizationTest2(test_file_contents, intervalIndexForSamples, encoded, quantizedSamples, quantizationError)
                else:
                    result = QuantizationTest1(test_file_contents, encoded, quantizedSamples)

                st.write(result)










