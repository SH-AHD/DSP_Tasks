import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from Tasks.task5.DerivativeSignal import DerivativeSignal

def read_input_file(file):
    """Reads signal data from uploaded file and returns the indices and values."""
    lines = file.readlines()
    num_samples = int(lines[2].strip())  # Number of samples
    indices = []
    samples = []

    for line in lines[3:3 + num_samples]:
        index, value = line.strip().split()
        samples.append(float(value))  # The signal value
        indices.append(int(index))  # The index

    return indices, samples


def fold_signal(signal):
    """Reverse the signal (fold it), maintaining positive values for amplitude."""
    folded_signal = signal[::-1]  # Reverse the signal
    folded_signal = [-x if x > 0 else x for x in folded_signal]  # Ensure positive values remain positive
    return folded_signal

def shift_signal(indices, signal, k):
    """Shifts the signal indices by k steps."""
    shifted_indices = [i + k for i in indices]  # Shift the indices
    shifted_signal = signal  # Keep the same signal values

    # Sort the result based on shifted indices for correct ordering
    sorted_pairs = sorted(zip(shifted_indices, shifted_signal))
    shifted_indices = [pair[0] for pair in sorted_pairs]
    shifted_signal = [pair[1] for pair in sorted_pairs]

    return shifted_indices, shifted_signal

def fold_and_shift(indices, signal, k):
    """Folds the signal and then shifts it by k steps."""
    # Step 1: Fold the signal (reverse the signal)
    folded_signal = fold_signal(signal)

    # Step 2: Shift the folded signal (Right shift for k > 0, Left shift for k < 0)
    shifted_indices, shifted_signal = shift_signal(indices, folded_signal, k)

    return shifted_indices, shifted_signal


def save_signal_to_file(signal, indices, filename):
    """Saves the signal data to a text file with the expected format."""
    with open(filename, 'w') as f:
        f.write("0\n0\n")
        f.write(f"{len(signal)}\n")
        for i, sample in zip(indices, signal):
            # Write the absolute value of the sample as an integer (remove decimal point)
            f.write(f"{i} {int(abs(sample))}\n")
 

def Shift_Fold_Signal(result_file, expected_file):
    """Compares the result file with the expected file."""
    # قراءة البيانات من ملف النتيجة
    with open(result_file, 'r') as f:
        # تجاهل الأسطر الثلاثة الأولى
        f.readline()  # 0
        f.readline()  # 0
        f.readline()  # عدد العينات

        result_indices = []
        result_samples = []
        for line in f.readlines():
            index, value = line.strip().split()
            result_indices.append(int(index))
            result_samples.append(int(float(value)))  # تحويل القيمة إلى عدد صحيح لتجاهل العلامة العشرية

    # قراءة البيانات من الملف المتوقع
    with open(expected_file, 'r') as f:
        # تجاهل الأسطر الثلاثة الأولى
        f.readline()  # 0
        f.readline()  # 0
        f.readline()  # عدد العينات

        expected_indices = []
        expected_samples = []
        for line in f.readlines():
            index, value = line.strip().split()
            expected_indices.append(int(index))
            expected_samples.append(int(float(value)))  # تحويل القيمة إلى عدد صحيح لتجاهل العلامة العشرية

    # مقارنة الأطوال
    if len(expected_samples) != len(result_samples) or len(expected_indices) != len(result_indices):
        return f"Test case failed, file lengths do not match. Expected: {len(expected_samples)} samples, Result: {len(result_samples)} samples."

    # مقارنة الفهارس
    for i in range(len(result_indices)):
        if result_indices[i] != expected_indices[i]:
            return f"Test case failed, indices do not match at index {i}. Expected index: {expected_indices[i]}, but got: {result_indices[i]}."

    # مقارنة القيم
    for i in range(len(result_samples)):
        if result_samples[i] != expected_samples[i]:
            return f"Test case failed, signal values do not match at index {i}. Expected value: {expected_samples[i]}, but got: {result_samples[i]}."

    return "Test case passed successfully"

 
# أفترض أن دوال مثل `shift_signal`, `fold_signal`, `fold_and_shift`, و `save_signal_to_file` موجودة.

def timedomapp():
    st.header("Time Domain Operations")

    # Session state for storing results
    if "result_signal" not in st.session_state:
        st.session_state.result_signal = None
    if "result_indices" not in st.session_state:
        st.session_state.result_indices = None

    select = st.selectbox("Select operation",
                          ["Compute Derivative Signal",
                           "Folding/Shifting",
                            ])

    if select == "Compute Derivative Signal":
        # Assuming DerivativeSignal is another part of the project
        DerivativeSignal()

    else:
        uploaded_file = st.file_uploader("Upload Signal File (txt format)", type=["txt"])

        if uploaded_file is not None:
            # Read signal data
            indices, signal = read_input_file(uploaded_file)
            if not signal:
                st.error("The uploaded signal file is empty or incorrectly formatted.")
                return

            st.write("### Original Signal:")
            # Display original signal graph
            plt.figure(figsize=(10, 4))
            plt.plot(indices, signal, label="Original Signal")
            plt.title("Original Signal in Time Domain")
            plt.xlabel("Samples")
            plt.ylabel("Amplitude")
            plt.grid(True)
            plt.legend()
            st.pyplot(plt)

            # Select operation
            operation = st.selectbox("Choose an operation:",
                                    ["Select", "Delay/Advance Signal", "Fold Signal", "Fold and Shift Signal"])

            if operation == "Delay/Advance Signal":
                k = st.number_input("Enter the number of steps (k):", min_value=-10000, step=1)
                action = st.radio("Choose an action:", ["Delay", "Advance"])

                if st.button("Apply Delay/Advance"):
                    if action == "Delay":
                        st.session_state.result_signal, st.session_state.result_indices = shift_signal(indices, signal, k)
                    else:
                        st.session_state.result_signal, st.session_state.result_indices = shift_signal(indices, signal, k)

                    # Display the result signal after shift/delay operation
                    st.write("### Processed Signal (Delay/Advance):")
                    plt.figure(figsize=(10, 4))
                    plt.plot(st.session_state.result_indices, st.session_state.result_signal, label="Processed Signal", color="orange")
                    plt.title("Processed Signal after Delay/Advance")
                    plt.xlabel("Samples")
                    plt.ylabel("Amplitude")
                    plt.grid(True)
                    plt.legend()
                    st.pyplot(plt)

            elif operation == "Fold Signal":
                if st.button("Apply Fold"):
                    st.session_state.result_signal = fold_signal(signal)
                    st.session_state.result_indices = indices

                    # Display the result signal after fold operation
                    st.write("### Processed Signal (Fold):")
                    plt.figure(figsize=(10, 4))
                    plt.plot(st.session_state.result_indices, st.session_state.result_signal, label="Folded Signal", color="green")
                    plt.title("Processed Signal after Fold")
                    plt.xlabel("Samples")
                    plt.ylabel("Amplitude")
                    plt.grid(True)
                    plt.legend()
                    st.pyplot(plt)

            elif operation == "Fold and Shift Signal":
                k = st.number_input("Enter the number of steps (k):", min_value=-10000, step=1)
                if st.button("Apply Fold and Shift"):
                    st.session_state.result_indices, st.session_state.result_signal = fold_and_shift(indices, signal, k)

                    # Display the result signal after fold and shift operation
                    st.write("### Processed Signal (Fold and Shift):")
                    plt.figure(figsize=(10, 4))
                    plt.plot(st.session_state.result_indices, st.session_state.result_signal, label="Folded and Shifted Signal", color="purple")
                    plt.title("Processed Signal after Fold and Shift")
                    plt.xlabel("Samples")
                    plt.ylabel("Amplitude")
                    plt.grid(True)
                    plt.legend()
                    st.pyplot(plt)

            # Save the resulting signal to a file
            if st.session_state.result_signal is not None and st.session_state.result_indices is not None:
                result_signal_filename = "result_signal.txt"
                save_signal_to_file(st.session_state.result_signal, st.session_state.result_indices, result_signal_filename)
                st.write(f"### Signal saved to file: {result_signal_filename}")

            # Compare with expected signal
            expected_file = st.file_uploader("Upload Expected Signal File (txt format)", type=["txt"], key="expected_file")

            if st.button("Compare with Expected Signal"):
                if expected_file is None:
                    st.error("Please upload an expected signal file for comparison.")
                elif st.session_state.result_signal is None or st.session_state.result_indices is None:
                    st.error("No resulting signal to compare. Perform an operation first.")
                else:
                    expected_file_path = "expected_signal.txt"
                    with open(expected_file_path, 'wb') as f:
                        f.write(expected_file.read())

                    comparison_result = Shift_Fold_Signal("result_signal.txt", expected_file_path)
                    st.write(comparison_result)



 