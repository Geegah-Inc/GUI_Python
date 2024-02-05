# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 20:59:26 2021

@author: Justin
"""


#ACQUISITION FUNCTIONS

def acqSingleFrameROI(xem, ADCNUM, file_name, c1 = 0, c2 = 127, c3 = 0, c4 = 127):
    import math
    
    xem.Open()
    xem.SelectADC(ADCNUM)
    xem.SelectFakeADC(0)
    xem.EnablePgen(0)
    xem.ResetFifo()
    xem.EnablePipeTransfer(1)
    xem.StartAcq()
    
    # Set the array size to match the data, but make it a multiple of 1024
    #print("1")
    nbytes = ((c2 - c1 + 1) * (c4 - c3 + 1))*2*2
    nbytes = 1024 * math.ceil(nbytes/1024)
    byte_data = bytearray(nbytes)
    nbytes = xem.GetPipeData(byte_data)
    #print("3")
    #print ("GetPipeData returned ", nbytes)
    f = open(file_name, "wb")
    f.write(byte_data)
    f.close()
    #print("Wrote data to roi.dat")
    #print("5")
    xem.Close()
    return byte_data

def acqSingleFrameCAL(xem, ADCNUM):
  
    xem.Open()
    xem.SelectADC(ADCNUM)
    xem.SelectFakeADC(0)
    xem.EnablePgen(0)
    xem.ResetFifo()
    xem.EnablePipeTransfer(1)
    xem.StartAcq()

    nbytes = 128*128*2*2
    byte_data = bytearray(nbytes)
    nbytes = xem.GetPipeData(byte_data)
    xem.Close()
    return byte_data


#SETTINGS FUNCTIONS

def reload_board(xem, frequency, roi_param):
    xem.Open()
    freq = frequency
    OUTEN = 1
    PSET =3 
    xem.SetROI(roi_param[0],roi_param[1],roi_param[2],roi_param[3])
    configureVCO(xem,freq,OUTEN,PSET)
    xem.Close()









def loadSavedRawDataRP(file_name):
    f = open(file_name, 'rb')
    MYDAT = f.read()
    f.close()
    I_RAW, Q_RAW = convertToIQImageRP(MYDAT)
    I_ADC, Q_ADC, I_VOLTS, Q_VOLTS = convertADCToVoltsRP(I_RAW, Q_RAW)
    return I_ADC, Q_ADC, I_VOLTS, Q_VOLTS

def impedance_si(ref_coef, array = True):
    
    csi = 8433
    psi = 2329
    zsi = csi*psi
    if array == False:
        zsamp = (zsi*(1-ref_coef))/(1+ref_coef)
        return zsamp/1e6
    if array ==True:
        new_list = []
        for jj in ref_coef:
            zsamp = (zsi*(1-jj))/(1+jj)
            new_list.append(zsamp/1e6)
        return new_list

def convertToIQImageRP(byte_data):
    import numpy as np
    wi = 0

    imgBytesI = np.zeros(128*128)
    imgBytesQ = np.zeros(128*128)
    for row in range (128):
        for col in range(128):
            wi = row*128 + col
            iwrd = (byte_data[4 * wi + 1] + 256*byte_data[4 * wi + 0]) #swap +0 and +1
            qwrd = (byte_data[4 * wi + 3] + 256*byte_data[4 * wi + 2]) #swap +2 and +3
            imgBytesI[wi] = iwrd
            imgBytesQ[wi] = qwrd
            
    IMG_I=imgBytesI.reshape([128,128])
    IMG_Q=imgBytesQ.reshape([128,128])
    return IMG_I, IMG_Q

def loadSavedRawDataROI(file_name, c1 = 0, c2 = 128, r1 = 0, r2 = 128):
    import numpy as np
    f = open(file_name, 'rb')
    MYDAT = f.read()
    f.close()
    rows = r2 - r1+1
    cols = c2 - c1+1
    imgBytesI = np.zeros(rows*cols)
    imgBytesQ = np.zeros(rows*cols)
    
    for row in range(rows):
        for col in range(cols):
            wi = row*cols + col  # Correction: use cols instead of rows
            iwrd = (MYDAT[4 * wi + 0] + 256*MYDAT[4 * wi + 1])
            qwrd = (MYDAT[4 * wi + 2] + 256*MYDAT[4 * wi + 3])
            imgBytesI[wi] = iwrd
            imgBytesQ[wi] = qwrd
            
    J_MYIMAGE_I = imgBytesI.reshape([rows, cols])
    J_MYIMAGE_Q = imgBytesQ.reshape([rows, cols])
    I_IMAGE_ADC = J_MYIMAGE_I / 16  # correct bit shift
    Q_IMAGE_ADC = J_MYIMAGE_Q / 16  # correct bit shift
    I_IMAGE_VOLTS = I_IMAGE_ADC * 1e-3  # convert to volts
    Q_IMAGE_VOLTS = Q_IMAGE_ADC * 1e-3  # convert to volts
    return I_IMAGE_ADC, Q_IMAGE_ADC, I_IMAGE_VOLTS, Q_IMAGE_VOLTS

    
def loadSavedRawDataFromBytes(bytes1):
    import numpy as np

    rows = 128
    cols = 128
    imgBytesI = np.zeros(rows*cols)
    imgBytesQ = np.zeros(rows*cols)
    
    for row in range(rows):
        for col in range(cols):
            wi = row*cols + col  # Correction: use cols instead of rows
            iwrd = (bytes1[4 * wi + 0] + 256*bytes1[4 * wi + 1])
            qwrd = (bytes1[4 * wi + 2] + 256*bytes1[4 * wi + 3])
            imgBytesI[wi] = iwrd
            imgBytesQ[wi] = qwrd
            
    J_MYIMAGE_I = imgBytesI.reshape([rows, cols])
    J_MYIMAGE_Q = imgBytesQ.reshape([rows, cols])
    I_IMAGE_ADC = J_MYIMAGE_I / 16  # correct bit shift
    Q_IMAGE_ADC = J_MYIMAGE_Q / 16  # correct bit shift
    I_IMAGE_VOLTS = I_IMAGE_ADC * 1e-3  # convert to volts
    Q_IMAGE_VOLTS = Q_IMAGE_ADC * 1e-3  # convert to volts
    return I_IMAGE_ADC, Q_IMAGE_ADC, I_IMAGE_VOLTS, Q_IMAGE_VOLTS



#save settings_file
def saveSettingsFile(savedirname,bit_file_name,freq,OUTEN,PSET,term_count,TX_SWITCH_EN_SETTINGS,PULSE_AND_SETTINGS,RX_SWITCH_EN_SETTINGS,GLOB_EN_SETTINGS,LO_CTRL_SETTINGS,ADC_CAP_SETTINGS,DAC_VOLTAGE,ADC_TO_USE):
    
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%m/%d/%Y %H:%M:%S")
    
    settings_file_name = savedirname + "settings.txt"
    
    # write to file
    f = open(settings_file_name, "w")
    f.write("date M/D/Y and time H/M/S = " + dt_string +"\n")
    
    f.write("Bit File Name: "+ bit_file_name +"\n") #write bit file name
    f.write("Frequency: "+str(freq)+" MHz"+"\n") #Frequency
    f.write("OUTEN: "+ str(OUTEN)+"\n")
    f.write("PSET: " + str(PSET)+"\n")
    f.write("Term Count: " + str(term_count)+"\n")
    f.write("TX_SWITCH_EN Settings: "+str(TX_SWITCH_EN_SETTINGS)+"\n")
    f.write("PULSE_AND Settings: " + str(PULSE_AND_SETTINGS)+"\n")
    f.write("RX_SWITCH_EN Settings: " + str(RX_SWITCH_EN_SETTINGS)+"\n")
    f.write("GLOB_EN Settings: " + str(GLOB_EN_SETTINGS)+"\n")
    f.write("LO_CTRL Settings: " + str(LO_CTRL_SETTINGS)+"\n")
    f.write("ADC_CAP Settings: " + str(ADC_CAP_SETTINGS)+"\n")
    f.write("DAC Voltage for All Pixels: " + str(DAC_VOLTAGE)+"\n")
    f.write("ADC Used: " + str(ADC_TO_USE)+"\n")
    
    f.close()
    print("Wrote data to " + settings_file_name)


#set up DAC to be the same for all pixels
def setAllPixSameDAC(xem,DAC_VOLTAGE):
    myDACVal = convertVoltToDAC(DAC_VOLTAGE)
    for row in range(128):
        print("loading DAC ROW (",row,") of 128")
        for col in range(128):
            for i_or_q in range(2):
                #print("loading DAC entry (",row,col,i_or_q,")") #if you want to print per pixel
                xem.LoadDACEntry(row, col, i_or_q, myDACVal) 
    print("Done loading DAC table")

#configure timing registers
def configTiming(xem,term_count,TX_SWITCH_EN_SETTINGS,PULSE_AND_SETTINGS,RX_SWITCH_EN_SETTINGS,GLOB_EN_SETTINGS,LO_CTRL_SETTINGS,ADC_CAP_SETTINGS):
    xem.Open()
    xem.SetTerminalCount(term_count) 
    xem.SetTiming(*((0,)+TX_SWITCH_EN_SETTINGS))   # TX_SWITCH_EN
    xem.SetTiming(*((1,)+PULSE_AND_SETTINGS) )   # PULSE_AND
    xem.SetTiming(*((2,)+RX_SWITCH_EN_SETTINGS))    # RX_SWITCH_EN
    xem.SetTiming(*((3,)+GLOB_EN_SETTINGS))    # GLOB_EN
    xem.SetTiming(*((4,)+LO_CTRL_SETTINGS))     # LO_CTRL
    xem.SetTiming(*((5,)+ADC_CAP_SETTINGS))  # ADC_CAPTURE #80 81
    xem.Close()

"""

freq ==> frequency in MHz --> resolution of 0.1MHz
OUTEN--> output enable-->0 for disable, 1 for enable
PSET --> select RF output power

power settings for PSET
0 --> Enabled, -4 dbm
1 --> Enabled, -1 dbm
2 --> Enabled, +2 dbm
3 --> Enabled, +5 dbm
don't use 2 or 3 unless really needed
"""
#configure VCO
def configureVCO(xem,freq,OUTEN,PSET):
    R0, R1, R2, R3, R4, R5 = calc_vco_reg_values(freq,OUTEN,PSET)
    #configure VCO registers on FPGA
    setXEMVCORegs(xem, R0, R1, R2, R3, R4, R5)

#configure VCO
def configureVCO_10khz(xem,freq,OUTEN,PSET):
    R0, R1, R2, R3, R4, R5 = calc_vco_reg_values_10khz(freq,OUTEN,PSET)
    #configure VCO registers on FPGA
    setXEMVCORegs(xem, R0, R1, R2, R3, R4, R5)
    
def configureVCO_10khz_fsweep(xem,freq,OUTEN,PSET):
    xem.Open()
    R0, R1, R2, R3, R4, R5 = calc_vco_reg_values_10khz(freq,OUTEN,PSET)
    #configure VCO registers on FPGA
    setXEMVCORegs(xem, R0, R1, R2, R3, R4, R5)
    xem.Close()
#convert raw ADC data to bit-shift corrected ADC data and convert to voltage
def convertADCToVolts(I_IMAGE, Q_IMAGE):
    I_IMAGE_ADC = I_IMAGE/16 #correct bit shift
    Q_IMAGE_ADC = Q_IMAGE/16 #correct bit shift
    I_IMAGE_VOLTS = I_IMAGE_ADC*1e-3 #convert to volts
    Q_IMAGE_VOLTS = Q_IMAGE_ADC*1e-3 #convert to volts
    return I_IMAGE_ADC, Q_IMAGE_ADC, I_IMAGE_VOLTS, Q_IMAGE_VOLTS


#convert raw ADC data to bit-shift corrected ADC data and convert to voltage
def convertADCToVoltsRP(I_IMAGE, Q_IMAGE):
    I_IMAGE_ADC = I_IMAGE/1 #correct bit shift
    Q_IMAGE_ADC = Q_IMAGE/1 #correct bit shift
    I_IMAGE_VOLTS = I_IMAGE_ADC*1e-3 #convert to volts
    Q_IMAGE_VOLTS = Q_IMAGE_ADC*1e-3 #convert to volts
    return I_IMAGE_ADC, Q_IMAGE_ADC, I_IMAGE_VOLTS, Q_IMAGE_VOLTS

#convert voltage to DAC value (for xem7305.bit)
def convertVoltToDAC(myVolt):
    import math
    myDACVal = (myVolt-2.9622)*(512/(-0.33))
    myDACVal = math.floor(myDACVal)
    return myDACVal

#set VCO registers
def setXEMVCORegs(xem, R0, R1, R2, R3, R4, R5 ):
    xem.SetRegField(xem.spi_wdata, R5)
    xem.SetRegField(xem.vco_spi_go, 1)
    xem.SetRegField(xem.spi_wdata, R4)
    xem.SetRegField(xem.vco_spi_go, 1)
    xem.SetRegField(xem.spi_wdata, R3)
    xem.SetRegField(xem.vco_spi_go, 1)
    xem.SetRegField(xem.spi_wdata, R2)
    xem.SetRegField(xem.vco_spi_go, 1)
    xem.SetRegField(xem.spi_wdata, R1)
    xem.SetRegField(xem.vco_spi_go, 1)
    xem.SetRegField(xem.spi_wdata, R0)
    xem.SetRegField(xem.vco_spi_go, 1)


"""

freq_des ==> frequency in MHz --> resolution of 0.1MHz
OUTEN--> output enable-->0 for disable, 1 for enable
PSET --> select RF output power

power settings for PSET
0 --> Enabled, -4 dbm
1 --> Enabled, -1 dbm
2 --> Enabled, +2 dbm
3 --> Enabled, +5 dbm
don't use 2 or 3 unless really needed
"""

def calc_vco_reg_values(freq_des_user,OUTEN_user,PSET_user):
    import math
    
    freq_des = freq_des_user
    OUTEN = OUTEN_user
    PSET = PSET_user
    #keep OUTEN in bounds
    if (OUTEN > 1):
        OUTEN = 0
    #keep PSET in bounds
    if (PSET > 3):
        PSET = 0
    
    #this is fixed, not user specifiable
    freq_res = 0.1 # in MHz
    
    #make sure it is in the correct resolution
    freq_actual = freq_res * round(freq_des / freq_res)
    
    #calculate divider setting
    myDIVSEL = math.ceil(math.log(math.ceil(2200/freq_actual))/math.log(2))
    myDIV = 2**(myDIVSEL);
    
    #reference oscillator
    refin = 10 # in MHz
    #VCO PFD (phase frequency detector) frequency
    myPFD = 10 # in MHz
    
    mylumped = freq_actual*myDIV/myPFD #N in the ADI software
    
    myINT = math.floor(mylumped)
    
    #mod goes from 2 to 4095
    #frac goes from 0 to mod-1
    #if 0 for frac, use 2 for mod
    
    #MOD = Refin/Fres
    #fdes_dec = freq_actual - math.floor(freq_actual)
    if ((mylumped - myINT) == 0):
        myMOD = 2
        myFRAC = 0
    else:
        myMOD = refin/freq_res
        myFRAC = myMOD * (mylumped-math.floor(mylumped))
        myMOD = round(myMOD) #if some decimal stuff left
        myFRAC = round(myFRAC)
        
        while(math.gcd(myMOD,myFRAC)>1):
            mygcd = math.gcd(myMOD,myFRAC)
            myMOD = round(myMOD/mygcd)
            myFRAC = round(myFRAC/mygcd)
        
    #structure of register 0
    #0 at db31, 16 bit INT, 12 bit FRAC, 3 control bits 000
    reg0 = (myFRAC*(2**3)) + (myINT*(2**15))
    #structure of register 1
    reg1 = 0x8008001  + (myMOD * (2**3))
    
    #calculate reg4
    reg4 = (2**23) #fundamental feedback select
    reg4 = reg4 + (myDIVSEL*(2**20)) #divider select
    reg4 = reg4 + (2**2) #control bits
    reg4 = reg4 + (PSET*(2**3)) #power setting
    reg4 = reg4 + (OUTEN*(2**5)) #output enable setting
    reg4 = reg4 + ((80)*(2**12)) #band select clock divider value
    #registers that are unchanged
    reg2 = 0x4E42
    reg3 = 0x4B3
    reg5 = 0x580005

    return reg0, reg1, reg2, reg3, reg4, reg5

def calc_vco_reg_values_10khz(freq_des_user,OUTEN_user,PSET_user):
    import math
    
    freq_des = freq_des_user
    OUTEN = OUTEN_user
    PSET = PSET_user
    #keep OUTEN in bounds
    if (OUTEN > 1):
        OUTEN = 0
    #keep PSET in bounds
    if (PSET > 3):
        PSET = 0
    
    #this is fixed, not user specifiable
    freq_res = 0.01 # in MHz
    
    #make sure it is in the correct resolution
    freq_actual = freq_res * round(freq_des / freq_res)
    
    #calculate divider setting
    myDIVSEL = math.ceil(math.log(math.ceil(2200/freq_actual))/math.log(2))
    myDIV = 2**(myDIVSEL);
    
    #reference oscillator
    refin = 10 # in MHz
    #VCO PFD (phase frequency detector) frequency
    myPFD = 10 # in MHz
    
    mylumped = freq_actual*myDIV/myPFD #N in the ADI software
    
    myINT = math.floor(mylumped)
    
    #mod goes from 2 to 4095
    #frac goes from 0 to mod-1
    #if 0 for frac, use 2 for mod
    
    #MOD = Refin/Fres
    #fdes_dec = freq_actual - math.floor(freq_actual)
    if ((mylumped - myINT) == 0):
        myMOD = 2
        myFRAC = 0
    else:
        myMOD = refin/freq_res
        myFRAC = myMOD * (mylumped-math.floor(mylumped))
        myMOD = round(myMOD) #if some decimal stuff left
        myFRAC = round(myFRAC)
        
        while(math.gcd(myMOD,myFRAC)>1):
            mygcd = math.gcd(myMOD,myFRAC)
            myMOD = round(myMOD/mygcd)
            myFRAC = round(myFRAC/mygcd)
        
    #structure of register 0
    #0 at db31, 16 bit INT, 12 bit FRAC, 3 control bits 000
    reg0 = (myFRAC*(2**3)) + (myINT*(2**15))
    #structure of register 1
    reg1 = 0x8008001  + (myMOD * (2**3))
    
    #calculate reg4
    reg4 = (2**23) #fundamental feedback select
    reg4 = reg4 + (myDIVSEL*(2**20)) #divider select
    reg4 = reg4 + (2**2) #control bits
    reg4 = reg4 + (PSET*(2**3)) #power setting
    reg4 = reg4 + (OUTEN*(2**5)) #output enable setting
    reg4 = reg4 + ((80)*(2**12)) #band select clock divider value
    #registers that are unchanged
    reg2 = 0x4E42
    reg3 = 0x4B3
    reg5 = 0x580005

    return reg0, reg1, reg2, reg3, reg4, reg5

#POST-PROCESSING
def imgvid_plot_IMG(matrix_list, directory, foldername, 
             vmin = 0.1, vmax = 3):
    import matplotlib.pyplot as plt
    import os
    import cv2
    """this function allows user to generate images, and video
    directly from the main matrices lists.
    matrix_list: List containing n number of frames. 
    directory: file path for saving these images
    foldername: A new folder of this string will be created in 
        the directory. This foldername string will also appear in
        the title
    vmin = lower value for the colorbar of the images
    vmax = upper value for the colorbar of the images
    
    """
    fig, ax = plt.subplots(figsize=(9, 9))
    savedirec = directory+foldername + '/'
    if not os.path.exists(savedirec):
        os.makedirs(savedirec)
    fps = 7
    img_array = []
    frame_title = 0
    for i in range(len(matrix_list)):
        ax.cla()
        out = matrix_list[i]

        pos = ax.imshow(out, vmin=vmin, vmax=vmax, cmap = 'hot', interpolation = 'bilinear')
        ax.set_title(foldername +" : Frame="+str(i))
        ax.set_xlabel("Columns")
        ax.set_ylabel("Rows")
        cbar = fig.colorbar(pos)

        plt.pause(0.01)
        img_name = savedirec+'/'+'frame'+str(frame_title)+'.png'
        fig.savefig(img_name)
        img = cv2.imread(img_name)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
        frame_title = frame_title + 1
        
        while i<len(matrix_list)-1:
            cbar.remove()
        
    vid_file_name = savedirec + foldername+'_'+'video.avi'
    out = cv2.VideoWriter(vid_file_name,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print("Done generating images and video from frames")
  

def imgvid_plot_fsweep(matrix_list, directory, foldername, 
             start_freq = 1600, end_freq = 2000,
             step_freq = 1,
             vmin = 0.1, vmax = 5):
    
    import matplotlib.pyplot as plt
    import math
    import os
    import cv2
    import numpy as np
    """this function allows user to generate images, and video
    directly from the main matrices lists.
    matrix_list: List containing n number of 128x128 frames. 
    directory: file path for saving these images
    foldername: A new folder of this string will be created in 
        the directory. This foldername string will also appear in
        the title
    start_freq = start frequency: appears in the title
    end_freq = end frequency
    step_frequency = the difference between two consecutive frequencies
        in the frequency sweep, also known as f_delta
    vmin = lower value for the colorbar of the images
    vmax = upper value for the colorbar of the images
    
    """
    fig, ax = plt.subplots(figsize=(9, 9))
    savedirec = directory+foldername + '/'
    if not os.path.exists(savedirec):
        os.makedirs(savedirec)
   
    fps = 7
    img_array = []
    frames_num = len(matrix_list[0])
    for myf in range(start_freq*100,end_freq*100,math.floor(step_freq*100)):
        f_counter = 0
        freq_title = np.round(myf/100,2)
        freq_folder = savedirec+'_FREQ_'+str(myf)+'/'
        if not os.path.exists(freq_folder):
            os.makedirs(freq_folder)
        
        for jj in range(frames_num):
            ax.cla()
            out = matrix_list[f_counter][jj]
    
            pos = ax.imshow(out, vmin=vmin, vmax=vmax, cmap = 'cividis', interpolation = 'bilinear')
            ax.set_title(foldername +" : Frequency="+str(freq_title)+'MHz, Frame ='+str(jj))
            ax.set_xlabel("Columns")
            ax.set_ylabel("Rows")
            cbar = fig.colorbar(pos)
    
            plt.pause(0.01)
            img_name = freq_folder+'/'+'freq'+str(freq_title)+'MHz_Frame '+str(jj)+'.png'
            fig.savefig(img_name)
            img = cv2.imread(img_name)
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)
            cbar.remove()
        vid_file_name = freq_folder+"VID_"+str(myf)+'MHz'+'.avi'
        out = cv2.VideoWriter(vid_file_name,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
        f_counter = f_counter+1

        freq_title = freq_title + step_freq    
    print("Done generating images and video from frames")

