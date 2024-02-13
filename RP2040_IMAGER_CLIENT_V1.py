# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 13:01:17 2023

Client program to run on local laptop
Must be connected to the Raspberry Pi Zero W Wifi network

Note that this script uses OPENCV!

Need to wait until Raspberry Pi has finished starting up and initializing the RP2040
I think this takes ~2 minutes

You can keep running this script without resetting host
host can just be powered up once

Based on the following tutorials:
https://www.digitalocean.com/community/tutorials/python-socket-programming-server-client
https://pythonprogramming.net/client-chatroom-sockets-tutorial-python-3/
https://pythonprogramming.net/buffering-streaming-data-sockets-tutorial-python-3/ 


@author: Justin Kuo
"""


#%% import libraries

#opencv is used to plot images faster than Matplotlib
import cv2
#for wifi communication
import socket
#for matrices
import numpy as np
#for sleep
import time
#for file writing
import os


#%% setup folder to write saved data to
#top directory where files will be stored

savedirname = "C:/Users/anujb/Downloads/RPB_test/" #change this to whatever you want your directory name to be
if not os.path.exists(savedirname):
    os.makedirs(savedirname)

#folder to dump raw .dat files in 
rawdata_save_dir = savedirname + "rawdata/"
if not os.path.exists(rawdata_save_dir):
    os.makedirs(rawdata_save_dir)
    

#%% Set up wifi

#this is the IP address of the Raspberry Pi Zero W
#this should never change
host ="192.168.4.1"
#port opened on the RPi for communication
port = 5560  # initiate port no above 1024
SIZE = 65536 #16384 #8192 

client_socket = socket.socket()  # instantiate
client_socket.connect((host, port))  # connect to the server

#commands
#command for doing 2 dummy reads to clear RP2040 buffer
cmd_dummy = 'senddmy'
#command for reading 1 set of I and Q frames
cmd_frame = 'sendfrm'
#command send to indicate done
#we don't use this in this script
cmd_done = 'zzzdone'

#%% send first dummy read to clear RP2040 buffers
print("Doing Dummy Read 1")
client_socket.send(cmd_dummy.encode())  #send command
data = client_socket.recv(SIZE)  # receive response
print(data.decode())
print("press the button")
#a = input("Press the Reset Button and Then Hit Enter") #note that this is not really needed after you do this once

#%% send second dummy read to clear RP2040 buffers
print("Doing Dummy Read 2")
client_socket.send(cmd_dummy.encode()) #send command
data = client_socket.recv(SIZE)  # receive response
print(data.decode())


#%% Miscellaneous functions
#for writing binary array to file
def writeFile(file_name, byte_data):
    # write to file
    f = open(file_name, "wb")
    f.write(byte_data)
    f.close()

#for reading a frame from the imager
def readFrame():
    #packet size
    exp_length = 65536
    # send command to take an image
    message = cmd_frame 
    # receive data back
    client_socket.send(message.encode()) 
    
    #for receiving frame data separated over multiple packets
    split_list = []
    #length of currently received data
    count = 0
    while (count < exp_length):
        data = client_socket.recv(SIZE)  # receive response
        datalen = len(data)
        count= count + datalen #update the number of received bytes
        message = 'rx_rcved ' + str(datalen).zfill(3) #acknowledge message
        #append received frame segments to list
        split_list.append(data)
        if (count <exp_length):
            client_socket.send(message.encode())  # send message
        
    #combine all the frame segments
    combined_bytes = split_list[0]
    for count in range(1,len(split_list)):
        combined_bytes = combined_bytes + split_list[count]
        
    #return the full frame
    return combined_bytes


#take raw byte_data and convert to ADC output (output is in ADC units)
def convertToIQImage(byte_data):
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




#%% read air data
print("Doing AIR READ")
combined_bytes = readFrame() #receive a frame
AIR_I, AIR_Q = convertToIQImage(combined_bytes) #separate I and Q out
air_file_name = rawdata_save_dir + "air_data.dat" #save binary data
writeFile(air_file_name, combined_bytes) #save binary data
print("Received Air Data")

#%% Read images 

#number of frames to image
numFrames = 20
numDigits = len(str(numFrames)) #this is for file save names
#128x128 pixels displays very small with OpenCV
#want to rescale up to a larger image size of 512 x 512 pixels
bitmap = np.zeros((512,512),np.uint8)
#initialize image window
cv2.imshow("Color Image", bitmap)
cv2.waitKey(100)
#time how long this takes
t1 = time.time()
for cf in range(numFrames):
    #read an image
    combined_bytes = readFrame()
    
    #frame_list.append(combined_bytes)
    F_IMG_I, F_IMG_Q = convertToIQImage(combined_bytes);
    
    #write file
    img_file_name = rawdata_save_dir+"frame"+str(cf).zfill(numDigits)+".dat"
    writeFile(img_file_name, combined_bytes) 
    #plot only Q
    # sub_I = F_IMG_I - AIR_I
    # I_MIN = 0
    # I_MAX = 40 
    # I_NORM = sub_I-I_MIN
    # I_NORM[I_NORM<0]=0
    # I_NORM[I_NORM>(I_MAX-I_MIN)]=I_MAX-I_MIN
    # I_NORM *= 255.0/(I_MAX-I_MIN)
    # I_NORM_IMG = I_NORM.astype(np.uint8)
    sub_Q = F_IMG_Q - AIR_Q
    Q_MIN = 0
    Q_MAX = 40 
    Q_NORM = sub_Q-Q_MIN
    Q_NORM[Q_NORM<0]=0
    Q_NORM[Q_NORM>(Q_MAX-Q_MIN)]=Q_MAX-Q_MIN
    Q_NORM *= 255.0/(Q_MAX-Q_MIN)
    Q_NORM_IMG = Q_NORM.astype(np.uint8)


    #rescale image using cubic interpolation (which is faster)
    img_rescaled = cv2.resize(Q_NORM_IMG,(512,512),interpolation=cv2.INTER_CUBIC) 
    #update bitmap
    bitmap[:,:] = img_rescaled
    #plot image
    cv2.imshow("Color Image", bitmap)
    cv2.waitKey(1) #nominal short delay --> can increase this time to get frame viewable for longer
    
#after finished imaging frames    
client_socket.close()  # close the connection
t2 = time.time() #end time
print("Took this long: ", t2-t1)
#prompt to close windows
#a = input("Enter to close windows")
cv2.destroyWindow("Color Image") #this sometimes returns an error - not sure why yet
cv2.destroyAllWindows()
