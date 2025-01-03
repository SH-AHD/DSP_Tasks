##task1.1 signal representation
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Function to read the signal file
def read_file(file):
    linesof_file = []
    samples = []

    # Read lines from the file
    for line in file.readlines():
        linesof_file.append(line.strip())

    # Determine the signal type and number of samples
    signaltype = int(linesof_file[0])
    periodic = int(linesof_file[1])
    num_samples = int(linesof_file[2])

    # Read the samples
    for l in range(3, 3 + num_samples):
        sample_data = list(map(float, linesof_file[l].split()))  # Convert text to decimal numbers
        samples.append(sample_data)

    return samples

# Function to represent the signal
def SignalRepresentation(samples):
    # Extract time and amplitude values
    x = np.array([sample[0] for sample in samples])
    y = np.array([sample[1] for sample in samples])

    # Create the plots
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    # Discrete signal representation
    ax[0].stem(x, y)
    ax[0].set_title("Discrete Signal Representation")
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")

    # Analog signal representation
    ax[1].plot(x, y)
    ax[1].axhline(0, color='black', linewidth=1)
    ax[1].set_title("Analog Signal Representation")
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("Y")

    # Adjust layout for better spacing
    plt.tight_layout()
    st.pyplot(fig)
