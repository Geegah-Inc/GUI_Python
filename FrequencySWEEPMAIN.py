
#frequencies in MHz
f_start = 1700
f_end = 1900
f_detla = 0.5


#SET THE DIRECTORY ACCORDING TO THE BUTTON
save_dir = "...path+exp name" + "Button name" + "Time stamp/" + 

#MAIN LOOP
xem.Open()
xem.SelectADC(0)
xem.SelectFakeADC(0)
xem.EnablePgen(0)
xem.Close()

for myf in range(f_start*100,f_end*100,math.floor(f_delta*100)):    
    f_to_use = myf/100

    geegah_hp.configureVCO_10khz_fsweep(xem,f_to_use,OUTEN,PSET)
    geegah_hp.configTiming(xem,term_count,TX_SWITCH_EN_SETTINGS,PULSE_AND_SETTINGS,
                          RX_SWITCH_EN_SETTINGS,GLOB_EN_SETTINGS,LO_CTRL_SETTINGS,ADC_CAP_SETTINGS)
    
    myf_file_name_echo = save_dir + "Frequecyecho" + str(f_to_use) +".dat"
    myf_data_echo = geegah_hp.acqSingleFrame_FSWEEP(xem, ADC_TO_USE, myf_file_name_echo)
    time.sleep(0.1) 
    geegah_hp.configTiming(xem,term_count_NE,TX_SWITCH_EN_SETTINGS_NE,PULSE_AND_SETTINGS_NE,
                          RX_SWITCH_EN_SETTINGS_NE,GLOB_EN_SETTINGS_NE,LO_CTRL_SETTINGS_NE,ADC_CAP_SETTINGS_NE)
    myf_file_name_noecho = save_dir + "Frequecynoecho" + str(f_to_use) +".dat"
    myf_base_data_ne = geegah_hp.acqSingleFrame_FSWEEP(xem, ADC_TO_USE, myf_file_name_noecho)                     
    time.sleep(0.1)
    #print("Currently at frequency: ", myf, " MHz")

xem.Close()
#print("Done Sweeping Baseline Frequencies, echo and no-echo")




#%%###################################### N FRAMES ############################## FSWEEP################


#SET THE DIRECTORY ACCORDING TO THE BUTTON
save_dir = "...path+exp name" + "Button name" + "Time stamp/" + 

#MAIN LOOP
xem.Open()
xem.SelectADC(0)
xem.SelectFakeADC(0)
xem.EnablePgen(0)
xem.Close()

for myf in range(f_start*100,f_end*100,math.floor(f_delta*100)):    
    f_to_use = myf/100

    geegah_hp.configureVCO_10khz_fsweep(xem,f_to_use,OUTEN,PSET)
    geegah_hp.configTiming(xem,term_count,TX_SWITCH_EN_SETTINGS,PULSE_AND_SETTINGS,
                          RX_SWITCH_EN_SETTINGS,GLOB_EN_SETTINGS,LO_CTRL_SETTINGS,ADC_CAP_SETTINGS)
    
    myf_file_name_echo = save_dir + "Frequecyecho" + str(f_to_use) +".dat"
    myf_data = geegah_hp.acqSingleFrame_FSWEEP(xem, ADC_TO_USE, myf_file_name_echo)
    time.sleep(0.1) 
    geegah_hp.configTiming(xem,term_count_NE,TX_SWITCH_EN_SETTINGS_NE,PULSE_AND_SETTINGS_NE,
                          RX_SWITCH_EN_SETTINGS_NE,GLOB_EN_SETTINGS_NE,LO_CTRL_SETTINGS_NE,ADC_CAP_SETTINGS_NE)
    myf_file_name_noecho = save_dir + "Frequecynoecho" + str(f_to_use) +".dat"
    myf_base_data_ne = geegah_hp.acqSingleFrame_FSWEEP(xem, ADC_TO_USE, myf_file_name_noecho)                     
    time.sleep(0.1)
    #print("Currently at frequency: ", myf, " MHz")

xem.Close()
#print("Done Sweeping Baseline Frequencies, echo and no-echo")
