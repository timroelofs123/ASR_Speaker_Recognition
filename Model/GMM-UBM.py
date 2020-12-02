import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score



#Import MFCC data
train_data=pd.read_csv(r'C:\Users\Alejandro\Documents\Master EIT Digital Data Science\Second Year - Aalto\Period 1\Speech recognition\Project\MFCC_data_train.csv',sep=',')
speaker_column=train_data.columns[-1]
speakers=np.unique(train_data[speaker_column])

test_data=pd.read_csv(r'C:\Users\Alejandro\Documents\Master EIT Digital Data Science\Second Year - Aalto\Period 1\Speech recognition\Project\MFCC_data_test.csv',sep=',')

#Create the dictionary of speakers, the list of GMMs and list of enrol datasets
dic_speakers={}
for i,speaker in enumerate(speakers):
    dic_speakers[i]=speaker
enrol_data_list=[]
GMM_list=[]

#Append the data from each speaker into their correspondin enroll dataset.
for speaker in speakers:
    enrol_data=train_data.loc[np.where(train_data[speaker_column]==speaker)[0]]
    enrol_data_list.append(enrol_data)


#Create the UMB with all of the training data
UBM=GaussianMixture(1024,covariance_type='diag')
UBM.fit(train_data.drop(columns=[speaker_column]).values)

#Create a GMM for each speaker and train it with 270 samples from their corresponding speaker
for i in range(len(enrol_data_list)):
    GMM = GaussianMixture(32, covariance_type='diag')
    GMM.fit(enrol_data_list[i].drop(columns=[speaker_column]).values[:270])
    GMM_list.append(GMM)

#Speaker Verification
real_speaker_list=np.array([])
predict_score_list=np.zeros((test_data.values.shape[0],speakers.shape[0]))
for i in range(test_data.values.shape[0]):
    UBM_score=UBM.score_samples(test_data.values[i][:int(speaker_column)].reshape(1,-1))[0] #Calculate the UMB score
    real_speaker = test_data.values[i][int(speaker_column)] #Obtain the real label
    real_speaker_list = np.append(real_speaker_list, real_speaker)
    for j in range(len(GMM_list)):
        # Calculate the speaker model score for each of the speaker models
        GMM_score=GMM_list[j].score_samples(test_data.values[i][:int(speaker_column)].reshape(1,-1))[0]
        #Calculate the likelihood ratio for each speaker model
        predict_score_list[i,j]=GMM_score-UBM_score
#We obtain the predictions as the model with highest likelihood ratio between all of the models
predictions=np.argmax(predict_score_list,axis=1)

#Obtain the predicted labels from their indexes
preds=np.array([])
for pred in predictions:
    preds=np.append(preds,dic_speakers[pred])

#Calculate the accuracy
accuracy=accuracy_score(real_speaker_list,preds)
print(accuracy)