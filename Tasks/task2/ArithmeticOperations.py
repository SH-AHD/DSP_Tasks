import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import itertools 
import operator 
from Tasks.task1.signalrepresentation import read_file


compareFile = None

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

def saveSignalToTxt(filename, n_values, signal_values):

    with open(filename, 'w') as file:
        file.write("0\n")  
        file.write("0\n") 
        N1 = len(signal_values)  
        file.write(f"{N1}\n")  

       
        for index, val in enumerate(signal_values):
            file.write(f"{index} {val:.6f}\n")  

        print(f"Signal saved to {filename}")



def add_signals(signals):
    lengths = [len(signal) for signal in signals]
    if len(signals) < 2:
        st.error("You must upload at least two signals for subtraction.")
        return None

    result = np.zeros_like([sample[1] for sample in signals[0]])
    for signal in signals:
        result += np.array([sample[1] for sample in signal])

    return list(zip(np.array([sample[0] for sample in signals[0]]), result))


def subtract_signals(signals):
    if len(signals) < 2:
        st.error("You must upload at least two signals for subtraction.")
        return None

    result = np.array([sample[1] for sample in signals[0]])
    for signal in signals[1:]:
        result -= np.array([sample[1] for sample in signal])

    return list(zip(np.array([sample[0] for sample in signals[0]]), result))

   

def multiplicationConstant (signal,constant) :
   
    constantXsignal=[(s[0],s[1]*constant) for s in signal]
    
    return constantXsignal

def squaringSignals(signal):
    signalSqr=[(s[0],s[1]**2) for s in signal]
    return signalSqr


def normalize_signal(signal, normalization_type):
    y_values = np.array([sample[1] for sample in signal])

    if normalization_type == "-1 to 1":
        normalized_values = 2 * (y_values - np.min(y_values)) / (np.max(y_values) - np.min(y_values)) - 1
    else:
        normalized_values = (y_values - np.min(y_values)) / (np.max(y_values) - np.min(y_values))

    return list(zip(np.array([sample[0] for sample in signal]), normalized_values))



def accumulationSignal(signal):
    y_values = np.array([sample[1] for sample in signal])
    accumulated_values = np.cumsum(y_values)
    return list(zip(np.array([sample[0] for sample in signal]), accumulated_values))
    # accResult =[(s[0] ,itertools.accumulate(s[1],operator.add)) for s in signal]
    # return accResult


def display_result_signal(result_signal,operation):
    st.header("Resulting Signal")
    x = np.array([sample[0] for sample in result_signal])
    y = np.array([sample[1] for sample in result_signal])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x[:20], y[:20])
    ax.set_title(f"Resulting Signal of {operation} Operation")
    ax.set_xlabel("Sample Index (n)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)

    plt.tight_layout()
    st.pyplot(fig)

          

def performArithmeticOperation():
    #global compareFile

    st.title("Arithmetic Operations on Signals")
    
    # File uploader for input signals
    uploaded_files = st.file_uploader("Upload multiple signal files", accept_multiple_files=True)
    
    if len(uploaded_files) == 0:
        st.warning("Please upload at least one signal file to perform operations.")
        return
   
    # Read input signals from uploaded files
    signals = []
    for file in uploaded_files:
        samples = read_file(file)
        signals.append(samples)
    
 
    #operation = st.selectbox("Choose an operation", ["Addition", "Subtraction", "Multiplication", "Squaring", "Normalization", "Accumulation"])
    st.text("Choose Operation :\n")
    add = sub = multiplication = Squaring = normalize = accumulation = False
    if  len(uploaded_files) > 1:    
        add= st.checkbox("Signals Addition ")
        sub = st.checkbox("Signals subtraction")
    if  len(uploaded_files) >= 1:
        multiplication = st.checkbox("Multiplication by constant")
        #scaling = st.checkbox("Scaling")
        Squaring= st.checkbox("Squaring")
        normalize = st.checkbox("Normalize")
        accumulation = st.checkbox("Accumulation")
   
   
    
     # Perform addition
    if add:
        result_signal = add_signals(signals)
        display_result_signal(result_signal, "Addition")
        saveSignalToTxt("addition_result.txt", [s[1] for s in result_signal])

        # Compare the result with expected file
        expected_file = st.file_uploader("Upload expected signal file for comparison", key="expected_addition")
        if expected_file:
            with open("expected_addition_signal.txt", "wb") as f:
                f.write(expected_file.getbuffer())
           # SignalSamplesAreEqualStreamlit("expected_addition_signal.txt", "addition_result.txt")

    # Perform subtraction
    if sub:
        result_signal = subtract_signals(signals)
        display_result_signal(result_signal, "Subtraction")
        saveSignalToTxt("subtraction_result.txt", [s[1] for s in result_signal])

        # Compare the result with expected file
        expected_file = st.file_uploader("Upload expected signal file for comparison", key="expected_subtraction")
        if expected_file:
            with open("expected_subtraction_signal.txt", "wb") as f:
                f.write(expected_file.getbuffer())
            #SignalSamplesAreEqualStreamlit("expected_subtraction_signal.txt", "subtraction_result.txt")

    if  multiplication:
        # signalname=st.selectbox("choose signal number: ",[n.name for n in uploaded_files])   
        
        # if signalname =="signal1":compareFile=r"Tasks\task2\Output files\MultiplySignalByConstant-Signal1 - by 5.txt"
        # elif signalname =="signal2":compareFile=r"Tasks\task2\Output files\MultiplySignalByConstant-Signal2 - by 10.txt"
        signal_number = st.selectbox("Choose signal number:", [f"Signal {i + 1}" for i in range(len(uploaded_files))])
        
        # Set the compareFile based on the selected signal
        if signal_number == "Signal 1":
            compareFile = r"Tasks\task2\Output files\MultiplySignalByConstant-Signal1 - by 5.txt"
        elif signal_number == "Signal 2":
            compareFile = r"Tasks\task2\Output files\MultiplySignalByConstant-Signal2 - by 10.txt"
    
        constant = st.number_input("Enter constant for multiplication", value=5)
        signal_index = int(signal_number.split()[1]) - 1  # Get the correct index
        result_signal = multiplicationConstant(signals[signal_index], constant)
        
        #result_signal = multiplicationConstant(signals[0], constant)
        display_result_signal(result_signal,"Multiplication")
        mfile=f"mby{constant},{signal_number}.txt"
        result_values = [s[1] for s in result_signal]
        saveSignalToTxt(mfile,result_signal[0],result_values)
        if compareFile:  # Check if compareFile is not None
         SignalSamplesAreEqual(compareFile,result_signal[0],result_values)
      

    if Squaring:
        signal_number = st.selectbox("Choose signal number:", [f"Signal {i + 1}" for i in range(len(uploaded_files))])
        signal_index = int(signal_number.split()[1]) - 1
        result_signal = squaringSignals(signals[signal_index])
        #result_signal = squaringSignals(signals[0])
        display_result_signal(result_signal,"Squaring")
        sfile=f"sqr{signal_number}.txt"
        result_values = [s[1] for s in result_signal]
        saveSignalToTxt(sfile,result_signal[0],result_values)
        
        compareFile=r"Tasks\task2\Output files\Output squaring signal 1.txt"
       
        SignalSamplesAreEqual(compareFile,result_signal[0],result_values)
      



  # Perform normalization
    if normalize:
        normalization_type = st.selectbox("Choose normalization type", ["-1 to 1", "0 to 1"], key="norm_type")
        signal_number = st.selectbox("Choose signal", [f"Signal {i + 1}" for i in range(len(uploaded_files))], key="norm_signal")
        signal_index = int(signal_number.split()[1]) - 1
        result_signal = normalize_signal(signals[signal_index], normalization_type)
        display_result_signal(result_signal, "Normalization")
        saveSignalToTxt(f"normalization_result.txt", [s[1] for s in result_signal])

        # Compare the result with expected file
        expected_file = st.file_uploader("Upload expected signal file for comparison", key="expected_normalization")
        if expected_file:
            with open("expected_normalization_signal.txt", "wb") as f:
                f.write(expected_file.getbuffer())
#            SignalSamplesAreEqualStreamlit("expected_normalization_signal.txt", "normalization_result.txt")

    if accumulation:
        signal_number = st.selectbox("Choose signal number:", [f"Signal {i + 1}" for i in range(len(uploaded_files))])
        signal_index = int(signal_number.split()[1]) - 1
        result_signal = accumulationSignal(signals[signal_index])      
       # result_signal = accumulationSignal(signals[0])
        display_result_signal(result_signal,"Accumulation")
        accfile=f"acc{signal_number}.txt"
        result_values = [s[1] for s in result_signal]
        saveSignalToTxt(accfile,result_signal[0],result_values)
        
        compareFile=r"Tasks\task2\Output files\output accumulation for signal1.txt"
       
        SignalSamplesAreEqual(compareFile,result_signal[0],result_values)
      

    
        


