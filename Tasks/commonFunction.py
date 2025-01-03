def saveSignalToTxt(filename, n_values, signal_values, amplitude, frequency, phase, sampling_frequency):

    with open(filename, 'w') as file:
        file.write("0\n")  
        file.write("0\n") 
        N1 = len(signal_values)  
        file.write(f"{N1}\n")  

       
        for index, val in enumerate(signal_values):
            file.write(f"{index} {val:.6f}\n")  

       # print(f"Signal with amplitude ={amplitude} , frequency={frequency} , phase={phase} , sampling_frequency={sampling_frequency} saved to {filename}")


