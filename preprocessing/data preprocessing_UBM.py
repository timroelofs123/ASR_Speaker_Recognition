import numpy as np
import os
import glob
import librosa
from sklearn.mixture import GaussianMixture
import soundfile as sf
import pandas as pd

test_files=open(r'C:\Users\Alejandro\Documents\Master EIT Digital Data Science\Second Year - Aalto\Period 1\Speech recognition\Project\test.txt').readlines()
train_files=open(r'C:\Users\Alejandro\Documents\Master EIT Digital Data Science\Second Year - Aalto\Period 1\Speech recognition\Project\train.txt').readlines()
for i in range(len(train_files)):
    train_files[i]=train_files[i].split('\n')[0]
for i in range(len(test_files)):
    test_files[i] = test_files[i].split('\n')[0]

speakers=np.array([]) #Speaker list
durations=np.array([]) #Duration of the recording list
MFCC_df_train=pd.DataFrame() #MFCC train dataframe
MFCC_df_test=pd.DataFrame() #MFCC test dataframe
for dir in (glob.glob(r'C:\Users\Alejandro\Documents\Master EIT Digital Data Science\Second Year - Aalto\Period 1\Speech recognition\Project\LibriSpeech_train\dev-clean\*', recursive=True)):
    #Iterate though each folder with the name of each speaker
    speaker=dir.split("\\")[-1] #obtain the speaker name
    for dir2 in glob.glob(dir+ "\*"): #iterate through each of the folders inside the speaker folder
        for file in glob.glob(dir2+ "\*.flac"): #iterate through each file that has the .flac extension
            filename=file.split("\\")[-1].split(".")[0] #get the same of the file
            x, sr = librosa.load(file) #load it to librosa
            mffcs = librosa.feature.mfcc(x, sr=sr) #extract the MFCC
            #sf.write(r'C:\Users\Alejandro\Documents\Master EIT Digital Data Science\Second Year - Aalto\Period 1\Speech recognition\Project\LibriSpeech_train_wav\dev-clean\\'+speaker+'\\'+filename+".wav",x,samplerate=sr)
            n_samples=mffcs.shape[1]//50 #Calculate the amount of samples to be extracted from that MFCC
            if(n_samples>0):
                for i in range(n_samples):
                    mffcs_sample=mffcs[:, i*50:i*50+50] #Cut a [20x50] sample from the MFCC
                    mffcs_sample=mffcs_sample.flatten() # flatten it
                    mffcs_sample=list(mffcs_sample) #make it a list
                    mffcs_sample.append(speaker) #Append the speaker label to it
                    mffcs_series=pd.Series(mffcs_sample) # Create a pandas series out of the list
                    if(filename in test_files):
                        # If the name of the file is in the test list, add it to the test dataframe
                        MFCC_df_test=MFCC_df_test.append(mffcs_series,ignore_index=True)
                    elif(filename in train_files):
                        # If the name of the file is in the train list, add it to the train dataframe
                        MFCC_df_train = MFCC_df_train.append(mffcs_series, ignore_index=True)
                    else:
                        #In case of not found anywhere print "Not found"
                        #This is just in case of mistake, in the final version it doesnt happen.
                        print('File not in training or testing txts')
#Save the dataframes into CVS files:
MFCC_df_test.to_csv(r'C:\Users\Alejandro\Documents\Master EIT Digital Data Science\Second Year - Aalto\Period 1\Speech recognition\Project\MFCC_data_test.csv',index = False)
MFCC_df_train.to_csv(r'C:\Users\Alejandro\Documents\Master EIT Digital Data Science\Second Year - Aalto\Period 1\Speech recognition\Project\MFCC_data_train.csv',index = False)



