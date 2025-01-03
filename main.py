import streamlit as st
from Tasks.task1.signalrepresentation import read_file, SignalRepresentation
from Tasks.task1.sin_cos import displaySignal
from Tasks.task2.ArithmeticOperations import performArithmeticOperation
from Tasks.task3.quantization import quantizationApp
from Tasks.task4.fourier import finalpresent
from Tasks.task5.timeDomain import timedomapp
from Tasks.task6.filters import process_signal
from Tasks.task7.FIR import fir_filter_ui
from Tasks.task7.resampling import resampling_application

st.set_page_config(page_title='DSP_Tasks', page_icon='📡')
st.title("Signal Processing Framework")  
    
   
menu = st.sidebar.selectbox("Choose an option", 
                            ["Signal Representation", 
                             "Generate Sinusoidal or Cosinusoidal Signals", 
                             "Arithmetic Operations", 
                             "Signal Quantization",
                             "Frequency Domain" ,
                             "Time Domain",
                             "Filters",
                             "FIR",
                             "Resampling"
                             ])

if menu == "Signal Representation":
     
        var = st.file_uploader(label='Upload your Signal file:')
        if var:
          
            samples = read_file(var)

          
            SignalRepresentation(samples)
elif menu == "generate sinusoidal or cosinusoidal signals":
        displaySignal()  
elif menu == "Arithmetic Operations":
        performArithmeticOperation()
elif menu== "Signal Quantization"  : 
        quantizationApp()

elif menu== "Frequency Domain" : 
       finalpresent()

elif menu=="Time Domain" :
       timedomapp()

elif menu== "Filters" :
       process_signal()       
elif menu=="FIR":
       fir_filter_ui()
elif menu=="Resampling":       
       resampling_application()
       