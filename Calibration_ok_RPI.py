# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 16:20:34 2024

@author: anujb
"""

#%% Import Libraries
import fpga #rainer functions
import sys
import numpy as np
import matplotlib.pyplot as plt
import geegah_hp
import time
import os

#%% Make directories  to save files in
foldername = "FSWEEP_TEST"
path = "C:/Users/anujb/Downloads"

savedirname = os.path.join(path, foldername, "")

if not os.path.exists(savedirname):
    os.makedirs(savedirname)
    
#folder to dump raw .dat files in for 1st acoustic echo data
rawdata_save_dir = savedirname + "rawdata_echo/"
if not os.path.exists(rawdata_save_dir):
    os.makedirs(rawdata_save_dir)

#folder to dump raw .dat files in for no echo
rawdata_ne_save_dir = savedirname + "rawdata_no_echo/"
if not os.path.exists(rawdata_ne_save_dir):
    os.makedirs(rawdata_ne_save_dir)
    
#folder to store images in
img_save_dir2 = savedirname + "images/"
if not os.path.exists(img_save_dir2):
    os.makedirs(img_save_dir2)

#folder to store video in
vid_save_dir = savedirname + "video/"
if not os.path.exists(vid_save_dir):
    os.makedirs(vid_save_dir)
    
#folder to store csv files in 
csv_save_dir = savedirname + "csv/"
if not os.path.exists(csv_save_dir):
    os.makedirs(csv_save_dir)

#folder to store baseline with echo files in 
BLE_save_dir = savedirname + "rawdata_baseline_echo/"
if not os.path.exists(BLE_save_dir):
    os.makedirs(BLE_save_dir)

#folder to store baseline with no echo files in 
BLNE_save_dir = savedirname + "rawdata_baseline_no_echo/"
if not os.path.exists(BLNE_save_dir):
    os.makedirs(BLNE_save_dir)

print("Done Setting Up Folders")

#%% Parameter selections

liveplot = True #boolean for plotting images real-time, True or False, set this as True for live plotting
frequency = 1853.5 #Pulse frequency in MHz, with resolution of 0.1 MHz

#Selection of firing/receiving pixels, ROI 
col_min = 0 #integer, 0<col_min<127
col_max = 127 #integer, 0<col_max<127
row_min = 0 #integer, 0<row_min<127
row_max = 127 #integer, 0<row_max<127

row_no = row_max - row_min
col_no = col_max - col_min
roi_param = [col_min, col_max, row_min, row_max]
num_Frames = 200 #Number of frames to acquire for sample, integer, num_Frames > 0

#%% FPGA setup 
#YOU ONLY HAVE TO RUN THIS ONCE AT THE BEGINING OF RUNNING THIS SCRIPT
#RE-RUN THIS IF YOU RESTARTED THE CONSOLE or RECONNECTED THE BOARD
#IF THIS IS ALREADY RAN ONCE, YOU ONLY NEED TO RUN THE reload_board() for quick settings load
xem = fpga.fpga()
board_name = xem.BoardName()
if board_name != "XEM7305":
    print("Problem: board name = " + board_name)  
    sys.exit()
print("Board: " + xem.di.deviceID + " " + xem.di.serialNumber)

#bit file to use (before DAC changes)
bit_file_name = "xem7305_GG222.bit"
xem.Configure(bit_file_name) # use older bit file
print("Version: " + xem.Version() + " serial number " + str(xem.SerialNumber()))
print("Sys clock = %8.4f MHz" % xem.SysclkMHz())

#setup vco
#frequency in MHz with resolution of 0.1MHz
#freq = 1883.7
freq = frequency
OUTEN = 1 #0 to disable, 1 to enable
#Power setting: 0 for -4 dbm, 1 for -1 dbm (2 for +2dbm, 3 for +5 dbm but don't use)
PSET = 3 #RF power setting
#configure VCO
geegah_hp.configureVCO(xem,freq,OUTEN,PSET)
# setup default timing
term_count = 83

#ROI setting######################################
xem.SetROI(col_min,col_max,row_min,row_max)
col_min = xem.GetRegField(xem.roi_start_col)
col_max = xem.GetRegField(xem.roi_end_col)
row_min = xem.GetRegField(xem.roi_start_row)
row_max = xem.GetRegField(xem.roi_end_row)
print("ROI register is (", col_min, col_max, row_min, row_max, ")")

#tuples to store timing settings
#(start logic state, I start, I end, Q start, Q end)
TX_SWITCH_EN_SETTINGS = (0, 19, 42, 19, 42) 
PULSE_AND_SETTINGS = (0, 20, 42, 20, 42) 
RX_SWITCH_EN_SETTINGS =( 0, 42, 82, 42, 82) 
GLOB_EN_SETTINGS = (1, 71, 1023, 71, 1023) 
LO_CTRL_SETTINGS = (1, 1023, 1023, 42, 1023) 
ADC_CAP_SETTINGS = (0, 80, 81, 80, 81)   # ADC_CAPTURE #80 81
#set the timing registers in the FPGA
#geegah_hp.configTiming(xem,term_count,TX_SWITCH_EN_SETTINGS,PULSE_AND_SETTINGS,RX_SWITCH_EN_SETTINGS,GLOB_EN_SETTINGS,LO_CTRL_SETTINGS,ADC_CAP_SETTINGS)

#terminal count Noecho
NE_del = 90
term_count_NE = 83+NE_del

#TIME SETTINGS FOR NO ECHO
#tuples to store timing settings
#(start logic state, I start, I end, Q start, Q end)
TX_SWITCH_EN_SETTINGS_NE = (0, 19, 42, 19, 42) 
PULSE_AND_SETTINGS_NE = (0, 20, 42, 20, 42) 
RX_SWITCH_EN_SETTINGS_NE =( 0, 42, 82+NE_del, 42, 82+NE_del) 
GLOB_EN_SETTINGS_NE = (1, 71+NE_del, 1023, 71+NE_del, 1023) 
LO_CTRL_SETTINGS_NE = (1, 1023, 1023, 42, 1023)
ADC_CAP_SETTINGS_NE = (0, 80+NE_del, 81+NE_del, 80+NE_del, 81+NE_del)   # ADC_CAPTURE #80 81

#set all DACs to the same voltage
#2.5V  works well
#don't go above 2.9V, probably best not to go below 1V
DAC_VOLTAGE = 2.8
geegah_hp.setAllPixSameDAC(xem,DAC_VOLTAGE) #comment this line out to skip setting the DACs

#ADC to use
#0 for gain of 5, 1 for no gain
ADC_TO_USE = 1
#Save settings to a file
geegah_hp.saveSettingsFile(savedirname,bit_file_name,freq,OUTEN,PSET,term_count,TX_SWITCH_EN_SETTINGS,PULSE_AND_SETTINGS,RX_SWITCH_EN_SETTINGS,GLOB_EN_SETTINGS,LO_CTRL_SETTINGS,ADC_CAP_SETTINGS,DAC_VOLTAGE,ADC_TO_USE)
#close the XEM (will be reopened later)
xem.Close()
time.sleep(0.05)
geegah_hp.configTiming(xem,term_count,TX_SWITCH_EN_SETTINGS,PULSE_AND_SETTINGS,RX_SWITCH_EN_SETTINGS,GLOB_EN_SETTINGS,LO_CTRL_SETTINGS,ADC_CAP_SETTINGS)
 
print("Done initializing FPGA")
#%% ONLY RUN THIS AFTER THE FPGA CODE SETUP HAVE BEEN RUN ONCE
geegah_hp.reload_board(xem, frequency, roi_param)
#%%
board = "OK"
#board = 'RP2040'
end_frequency = 1950
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

adjusted_i, adjusted_q, calibration_params = calibrate_iq_signals(i, q)

plt.plot(freqs,adjusted_i, label = 'I', linewidth = 3)
plt.plot(freqs, adjusted_q, label = 'Q', linewidth = 3)
plt.title("Adjusted I and Q vs frequency (64,64)")
plt.ylabel("Adjusted Echo (V)")
plt.xlabel("Frequency (MHz)")
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

# Usage:
#%%
#TEST ON A FRAME: 
#AIR FRAME
myf_air_data = geegah_hp.acqSingleFrameCAL(xem, ADC_TO_USE)
I_ADC,Q_ADC,I_A_VOLTS,Q_A_VOLTS = geegah_hp.loadSavedRawDataFromBytes(myf_air_data)
MAG_A = np.sqrt(np.square(I_A_VOLTS)+np.square(Q_A_VOLTS))
#%% 
#SAMPLE FRAME
myf_sample_data = geegah_hp.acqSingleFrameCAL(xem, ADC_TO_USE)
I_ADC,Q_ADC,I_S_VOLTS,Q_S_VOLTS = geegah_hp.loadSavedRawDataFromBytes(myf_sample_data) 
MAG_S = np.sqrt(np.square(I_S_VOLTS)+np.square(Q_S_VOLTS))
#%% 
DIFF_I = I_S_VOLTS - I_A_VOLTS
plt.imshow(DIFF_I, vmin = -0.01, vmax = 0.01, cmap = 'pink')
plt.colorbar()
plt.show()
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.title("I echo: baseline adj")
#%%
DIFF_Q = Q_S_VOLTS - Q_A_VOLTS
plt.imshow(DIFF_Q, vmin = -0.01, vmax = 0.01, cmap = 'pink')
plt.colorbar()
plt.show()
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.title("Q echo: baseline adj")
#%%
DIFF_MAG = MAG_S - MAG_A
plt.imshow(DIFF_MAG, vmin = -0.01, vmax = 0.01, cmap = 'pink')
plt.colorbar()
plt.show()
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.title("Magnitdue: baseline adj")


#%%
I_S_new, Q_S_new = adjust_iq_arrays(I_S_VOLTS, Q_S_VOLTS, calibration_params)
MAG_S_new = np.sqrt(np.square(I_S_new)+np.square(Q_S_new))

I_A_new, Q_A_new = adjust_iq_arrays(I_A_VOLTS, Q_A_VOLTS, calibration_params)
MAG_A_new = np.sqrt(np.square(I_A_new)+np.square(Q_A_new))
#%% 
DIFF_I = np.array(I_S_new) - np.array(I_A_new)
plt.imshow(DIFF_I, vmin = -0.01, vmax = 0.01, cmap = 'pink')
plt.colorbar()
plt.show()
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.title("I echo: baseline adj")
#%%
DIFF_Q = np.array(Q_S_new) - np.array(Q_A_new)
plt.imshow(DIFF_Q, vmin = -0.01, vmax = 0.01, cmap = 'pink')
plt.colorbar()
plt.show()
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.title("Q echo: baseline adj")
#%%
DIFF_MAG = np.array(MAG_S) - np.array(MAG_A)
plt.imshow(DIFF_MAG, vmin = -0.01, vmax = 0.01, cmap = 'pink')
plt.colorbar()
plt.show()
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.title("Magnitdue: baseline adj")
#%%WORKING WITH MATRICES
import numpy as np

def calibrate_iq_pixelwise(i_matrices, q_matrices):
    # Check the first matrix to get the number of rows and columns (assuming all matrices have the same shape)
    rows, cols = i_matrices[0].shape

    # Initialize matrices to hold calibration parameters
    scale_matrix = np.zeros((rows, cols))
    shift_matrix = np.zeros((rows, cols))
    adjust_i_matrix = np.zeros((rows, cols), dtype=bool)

    # Iterate over each pixel to find calibration parameters
    for x in range(rows):
        for y in range(cols):
            # Collect I and Q values for this pixel across all frequencies
            i_values = np.array([i_matrix[x, y] for i_matrix in i_matrices])
            q_values = np.array([q_matrix[x, y] for q_matrix in q_matrices])

            # Calculate basic statistics for I and Q
            i_max, i_min = i_values.max(), i_values.min()
            q_max, q_min = q_values.max(), q_values.min()

            i_mid = (i_max + i_min) / 2
            q_mid = (q_max + q_min) / 2

            i_range = i_max - i_min
            q_range = q_max - q_min

            # Decide which to adjust and calculate scale and shift
            adjust_i = q_range > i_range
            adjust_i_matrix[x, y] = adjust_i
            scale = q_range / i_range if adjust_i else i_range / q_range
            shift = q_mid - i_mid if adjust_i else i_mid - q_mid

            # Store calibration parameters
            scale_matrix[x, y] = scale
            shift_matrix[x, y] = shift

    # Apply calibration to each frequency matrix
    adjusted_i_matrices = []
    adjusted_q_matrices = []
    for i_matrix, q_matrix in zip(i_matrices, q_matrices):
        adjusted_i_matrix = np.zeros_like(i_matrix)
        adjusted_q_matrix = np.zeros_like(q_matrix)
        for x in range(rows):
            for y in range(cols):
                scale = scale_matrix[x, y]
                shift = shift_matrix[x, y]
                adjust_i = adjust_i_matrix[x, y]
                if adjust_i:
                    adjusted_i_matrix[x, y] = i_matrix[x, y] * scale + shift
                    adjusted_q_matrix[x, y] = q_matrix[x, y]
                else:
                    adjusted_q_matrix[x, y] = q_matrix[x, y] * scale + shift
                    adjusted_i_matrix[x, y] = i_matrix[x, y]
        adjusted_i_matrices.append(adjusted_i_matrix)
        adjusted_q_matrices.append(adjusted_q_matrix)

    return adjusted_i_matrices, adjusted_q_matrices, scale_matrix, shift_matrix, adjust_i_matrix

# Usage:
# i_matrices and q_matrices are lists of matrices, each matrix representing I or Q data for a specific frequency.
adjusted_i_matrices, adjusted_q_matrices, scale_matrix, shift_matrix, adjust_i_matrix = calibrate_iq_pixelwise(i_mat, q_mat)

#%%
import numpy as np

def adjust_new_iq_matrices(new_i_matrix, new_q_matrix, scale_matrix, shift_matrix, adjust_i_matrix):
    # Check the dimensions of the new matrices
    rows, cols = new_i_matrix.shape

    # Initialize matrices to hold adjusted values
    adjusted_new_i_matrix = np.zeros_like(new_i_matrix)
    adjusted_new_q_matrix = np.zeros_like(new_q_matrix)

    # Iterate over each pixel
    for x in range(rows):
        for y in range(cols):
            # Extract calibration parameters for this pixel
            scale = scale_matrix[x, y]
            shift = shift_matrix[x, y]
            adjust_i = adjust_i_matrix[x, y]

            # Apply adjustment based on the calibration parameters
            if adjust_i:
                adjusted_new_i_matrix[x, y] = new_i_matrix[x, y] * scale + shift
                adjusted_new_q_matrix[x, y] = new_q_matrix[x, y]  # Q remains unchanged
            else:
                adjusted_new_q_matrix[x, y] = new_q_matrix[x, y] * scale + shift
                adjusted_new_i_matrix[x, y] = new_i_matrix[x, y]  # I remains unchanged

    return adjusted_new_i_matrix, adjusted_new_q_matrix

# Usage example:
# new_i_matrix and new_q_matrix are the new matrices that you want to adjust.
# scale_matrix, shift_matrix, adjust_i_matrix are the matrices of calibration parameters obtained from the calibration process.
ADJ_I_S, ADJ_Q_S = adjust_new_iq_matrices(I_S_VOLTS, Q_S_VOLTS, scale_matrix, shift_matrix, adjust_i_matrix)
ADJ_I_A, ADJ_Q_A = adjust_new_iq_matrices(I_A_VOLTS, Q_A_VOLTS, scale_matrix, shift_matrix, adjust_i_matrix)
ADJ_MAG_A = np.sqrt(np.square(ADJ_I_A)+np.square(ADJ_Q_A))
ADJ_MAG_S = np.sqrt(np.square(ADJ_Q_S) + np.square(ADJ_I_S))
#%%

DIFF_I = np.array(ADJ_I_S) - np.array(ADJ_I_A)
plt.imshow(DIFF_I, vmin = -0.01, vmax = 0.01, cmap = 'pink')
plt.colorbar()
plt.show()
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.title("I echo: baseline adj")
#%%
DIFF_Q = np.array(ADJ_Q_S) - np.array(ADJ_Q_A)
plt.imshow(DIFF_Q, vmin = -0.01, vmax = 0.01, cmap = 'pink')
plt.colorbar()
plt.show()
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.title("Q echo: baseline adj")
#%%
DIFF_MAG = np.array(ADJ_MAG_S) - np.array(ADJ_MAG_A)
plt.imshow(DIFF_MAG, vmin = -0.01, vmax = 0.01, cmap = 'pink')
plt.colorbar()
plt.show()
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.title("Magnitdue: baseline adj")

#%%

import numpy as np

def rms_calc(frames):
    # Center the signal around zero
    mean_value = np.mean(frames)
    fluctuations = frames - mean_value
    
    # Calculate the RMS of these fluctuations
    rms = np.sqrt(np.mean(np.square(fluctuations)))
    return rms

def calculate_signal_quality(air_data, water_data):
    # Calculate the 'true' signal as the mean of the absolute difference between water and air
    true_signal = np.abs(water_data - air_data)
    signal_mean = np.mean(true_signal)
    # Calculate noise as the RMS of the fluctuations in the air data
    noise_rms = rms_calc(air_data)
    std1 = np.std(air_data)
    # Calculate SNR using the mean of the true signal and the RMS of the noise
    snr = 20 * np.log(signal_mean / 2*noise_rms) if noise_rms != 0 else float('inf')
    snr = (signal_mean)**2/std1
    
    # Calculate additional metrics
    mse = np.mean((true_signal - air_data) ** 2)
    rmse = np.sqrt(mse)
    #correlation_coefficient = np.corrcoef(air_data.flatten(), true_signal.flatten())[0, 1
    return {
        'SignalMean': signal_mean,
        'NoiseRMS': noise_rms,
        'SNR': snr,
        'MSE': mse,
        'RMSE': rmse,
        #'CorrelationCoefficient': correlation_coefficient
    }


# air_data is your air matrix and water_data is your water matrix.
signal_quality_metrics = calculate_signal_quality(np.array(ADJ_I_A),np.array(ADJ_I_S))
print("Signal Quality Metrics:", signal_quality_metrics)

signal_quality_metrics = calculate_signal_quality(np.array(I_A_new), np.array(I_S_new))
print("Signal Quality Metrics:", signal_quality_metrics)

signal_quality_metrics = calculate_signal_quality(np.array(I_A_VOLTS), np.array(I_S_VOLTS))
print("Signal Quality Metrics:", signal_quality_metrics)
#%%

# air_data is your air matrix and water_data is your water matrix.
signal_quality_metrics = calculate_signal_quality(np.array(ADJ_MAG_A),np.array(ADJ_MAG_S))
print("Signal Quality Metrics:", signal_quality_metrics)

signal_quality_metrics = calculate_signal_quality(np.array(MAG_A_new), np.array(MAG_S_new))
print("Signal Quality Metrics:", signal_quality_metrics)

signal_quality_metrics = calculate_signal_quality(np.array(MAG_A), np.array(MAG_S))
print("Signal Quality Metrics:", signal_quality_metrics)
#%%

#plot setup
plt.ion()
fig, ax = plt.subplots()
line1, = ax.plot([], [], label='I echo')  # Line for I values
line2, = ax.plot([], [], label='Q echo')  # Line for Q values
ax.set_xlabel('Frame')
ax.set_ylabel('Average Echo (V)')
ax.legend()
# lims
frames = 200
ax.set_xlim(0, frames) 
#ax.set_ylim(1,3)  
i_av = []
q_av = []
for frames in range(0,frames,1):
    freqs.append(frames)
    #OPAL KELLY
    if board == 'OK':
        myf_meas_data = geegah_hp.acqSingleFrameCAL(xem, ADC_TO_USE)
        I_ADC,Q_ADC,I_VOLTS,Q_VOLTS = geegah_hp.loadSavedRawDataFromBytes(myf_meas_data)

        ADJ_I_S, ADJ_Q_S = adjust_new_iq_matrices(I_VOLTS, Q_VOLTS, scale_matrix, shift_matrix, adjust_i_matrix)
       
        i_av.append(np.mean(ADJ_I_S.flatten()))
        q_av.append(np.mean(ADJ_Q_S.flatten()))
    
        # Set new data for the lines
        line1.set_xdata(frames)
        line1.set_ydata(i_av)
        line2.set_xdata(frames)
        line2.set_ydata(q_av)
    
        # Adjust the plot limits
        ax.set_xlim(0, frames + 1)  # Update x-axis to include new frame
        ax.set_ylim(min(i_av + q_av), max(i_av + q_av))  # Update y-axis to include new data range
    
        # Redraw the plot
        fig.canvas.draw()
        fig.canvas.flush_events()

plt.ioff()  # Turn off interactive mode
plt.show()
