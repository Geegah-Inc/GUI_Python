# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:06:56 2024

@author: anujb
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 16:20:34 2024

@author: anujb
"""

"""
=> Calibration: Initialization of the board
                - frequency sweep
                - calibrate_iq_signals (i,q), i, q = 1 D array => IMP calibration param, f_i_equal_q
                - Set frequency to f_i_equal_q: geegah_hp.configureVCO_10khz_fsweep(xem,f_i_equal_q,OUTEN,PSET)
                
                => Acquire air data:  1st AIR DATA AS A PART OF CALIBRATION
                
                    - Air echo: air echo bytes data returned
                    - Air no-echo:  air echo bytes data returned
                
                    -Iadc, q adc, I volts air, Q VOLTS air  = justin function, geegah_hp.loadSavedRawDataFromBytes()
                    -I_air adjusted, q_air adjusted = adjust_iq_arrays(I volts air, Q VOLTS air, calibration params)
                    -SAVING:  I adjusted air, Q adjusted air


=> MAIN LOOP

    -> Sample echo:  echo meas dat file
    -> Sample no-echo: save no-echo non adjusted
    
    I_volts sample, Q_volts sample =____
    I_volts sample adjusted, Q_volts sample adjusted = adjust_iq_arrays(I volts sample, Q VOLTS sample, calibration params)
    
    
    save I adjusted sample, save Q adjusted sample
    
    
    if IQ plot:
        I_plot = I adjusted sample - I adjusted air
        Q_plot = Q adjusted sample - I adjusted air
        
    if Mag/Phase:
        calculate mag=/phase using the adjusted I/Q (air, sample)
            
        
->Retaking air data process:
    - latest echo.dat, no-echo dat for the baseline gets updated
    - Calculate adjusted I and Q for air 
            
"""
#%%
board = "OK"
#board = 'RP2040'

end_frequency = 1860
start_frequency = 1800
frequency_interval = 0.5

#CALIBRATION FREQUENCY SWEEP
import math
#opal kelly frequency set
def f_set_OK(xem,frequency, OUTEN, PSET):
    geegah_hp.configureVCO_10khz_fsweep(xem,frequency,OUTEN,PSET)
    
#frequency sweep
i,q,freqs = [],[],[]
#plot setup
plt.ion()
fig, ax = plt.subplots()
line1, = ax.plot([], [], label='I echo')  # Line for I values
line2, = ax.plot([], [], label='Q echo')  # Line for Q values
ax.set_xlabel('Frequency (MHz)')
ax.set_ylabel('Echo (V)')
ax.legend()
# lims
ax.set_xlim(start_frequency, end_frequency) 
ax.set_ylim(1,3)  
i_mat = []
q_mat = []
for freq in range(start_frequency*100,end_frequency*100,math.floor(frequency_interval*100)):
    freq = freq/100
    freqs.append(freq) 
    
    #OPAL KELLY
    if board == 'OK':
        
        f_set_OK(xem,freq,OUTEN,PSET) #SWITCH FREQUENCY
        myf_meas_data = geegah_hp.acqSingleFrameCAL(xem, ADC_TO_USE)
        I_ADC,Q_ADC,I_VOLTS,Q_VOLTS = geegah_hp.loadSavedRawDataFromBytes(myf_meas_data)
        i.append(I_VOLTS[1,1])
        q.append(Q_VOLTS[1,1])
        i_mat.append(I_VOLTS)
        q_mat.append(Q_VOLTS)
        
        # Set new data for the lines
        line1.set_xdata(freqs)
        line1.set_ydata(i)
        line2.set_xdata(freqs)
        line2.set_ydata(q)
        
        # Adjust the plot limits
        ax.relim()
        ax.autoscale_view()
        # Redraw the plot
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)
        
plt.ioff()  # Turn off interactive mode
plt.show()     

#%% FINDING THE LARGEST FREQUENCY WHERE I = Q

def find_largest_magnitude_frequency(i_adj, q_adj, freqs):
    # Calculate the absolute difference between I and Q
    diff = np.abs(np.array(i_adj) - np.array(q_adj))
    # Indices where I ~ Q
    equal_indices = np.where(diff <= 0.01)[0]
    # Find the frequency where the magnitude of I or Q is the largest
    signal = np.maximum(np.abs(np.array(i_adj)[equal_indices]), np.abs(np.array(q_adj)[equal_indices]))
    max_signal_index = equal_indices[np.argmax(signal)]
    # Corresponding frequency
    I_equal_q_freq = freqs[max_signal_index]
    return I_equal_q_freq
  
#%%
def calibrate_iq_signals(i, q):
    # Calculate basic statistics: max, min, midpoint, and range for both signals
    i_max, i_min = max(i), min(i)
    q_max, q_min = max(q), min(q)

    i_mid = (i_max + i_min) / 2
    q_mid = (q_max + q_min) / 2

    i_range = i_max - i_min
    q_range = q_max - q_min

    # Decide which signal to adjust based on range
    adjust_i = q_range > i_range

    # Calculate scale and shift
    if adjust_i:
        scale = q_range / i_range
        shift = q_mid - i_mid
        adjusted_i = [i_mid + (x - i_mid) * scale + shift for x in i]
        adjusted_q = q
    else:
        scale = i_range / q_range
        shift = i_mid - q_mid
        adjusted_q = [q_mid + (x - q_mid) * scale + shift for x in q]
        adjusted_i = i

    # Calculate new statistics after adjustment
    new_i_max, new_i_min = max(adjusted_i), min(adjusted_i)
    new_q_max, new_q_min = max(adjusted_q), min(adjusted_q)
    new_i_mid = (new_i_max + new_i_min) / 2
    new_q_mid = (new_q_max + new_q_min) / 2

    # Return adjusted signals and calibration parameters
    calibration_params = {
        "scale": scale,
        "shift": shift,
        "adjusted_i_mid": new_i_mid,
        "adjusted_q_mid": new_q_mid
    }
    return adjusted_i, adjusted_q, calibration_params

# Usage:
i = i
q = q

# After adjusting the signals
adjusted_i, adjusted_q, calibration_params = calibrate_iq_signals(i, q)
# f where I equals Q
frequency = find_largest_magnitude_frequency(adjusted_i, adjusted_q, freqs)

# Plotting
plt.plot(freqs, adjusted_i, label='I', linewidth=3)
plt.plot(freqs, adjusted_q, label='Q', linewidth=3)

# Highlight the point with the largest magnitude where I equals Q
IQ_idx = freqs.index(frequency)
IQ_val = adjusted_i[IQ_idx]

plt.scatter([frequency],[IQ_val] , color='blue', s=100, zorder=5, label = 'I = Q frequency')

plt.title("Adjusted I and Q vs frequency (64,64)")
plt.ylabel("Adjusted Echo (V)")
plt.legend()
plt.show()

#%%
def adjust_iq_arrays(new_i_array, new_q_array, calibration_params):
    scale = calibration_params['scale']
    shift = calibration_params['shift']
    adjusted_i_mid = calibration_params['adjusted_i_mid']
    adjusted_q_mid = calibration_params['adjusted_q_mid']

    # Initialize lists to hold adjusted values
    adjusted_new_i_array = []
    adjusted_new_q_array = []

    # Adjust each element in the I and Q arrays
    if 'adjusted_i_mid' in calibration_params:  # This means I was adjusted in the calibration
        for new_i, new_q in zip(new_i_array, new_q_array):
            adjusted_new_i = adjusted_i_mid + (new_i - adjusted_i_mid) * scale + shift
            adjusted_new_q = new_q  # Q remains the same as it wasn't adjusted
            adjusted_new_i_array.append(adjusted_new_i)
            adjusted_new_q_array.append(adjusted_new_q)
    else:  # This means Q was adjusted in the calibration
        for new_i, new_q in zip(new_i_array, new_q_array):
            adjusted_new_q = adjusted_q_mid + (new_q - adjusted_q_mid) * scale + shift
            adjusted_new_i = new_i  # I remains the same as it wasn't adjusted
            adjusted_new_i_array.append(adjusted_new_i)
            adjusted_new_q_array.append(adjusted_new_q)

    return adjusted_new_i_array, adjusted_new_q_array


ADJ_I, ADJ_Q = adjust_iq_arrays(I_S_VOLTS,Q_S_VOLTS, calibration_params)


