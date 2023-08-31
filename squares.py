import numpy as np
import matplotlib.pyplot as plt

# Parameters
frequency = 1.0  # Frequency of the square wave (in Hz)
amplitude = 1.0  # Amplitude of the square wave
duration = 10.0   # Duration of the signal (in seconds)
sampling_rate = 1000  # Number of samples per second

# Generate time values
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# Generate the square signal
square_signal = amplitude * np.sign(np.sin(t))

# Plot the square signal
plt.plot(t, square_signal)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Square Signal')
plt.ylim(-1.5, 1.5)
plt.grid(True)
plt.show()