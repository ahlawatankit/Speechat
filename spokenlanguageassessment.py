import sys
def my_except_hook(exctype, value, traceback):
        print('There has been an error in the system')
#sys.excepthook = my_except_hook
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import parselmouth
from parselmouth.praat import call, run_file
import glob
import errno
import csv,sys
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import os
from subprocess import check_output
from sklearn import preprocessing
import queue
import soundfile as sf
import _thread  
import pickle
from scipy.stats import binom
from scipy.stats import ks_2samp
from scipy.stats import ttest_ind
from pandas import read_csv



audioFilesPath="dataset"+"/"+"vahan-TC-20-Dataset"+"/"    # Path for audio-files

pa1="dataset"+"/"+"datanewchi22.csv"
pa2="dataset"+"/"+"stats.csv"
pa3="dataset"+"/"+"datacorrP.csv"
pa4="dataset"+"/"+"datanewchi.csv"
pa5="dataset"+"/"+"datanewchi33.csv"
pa6="dataset"+"/"+"datanewchi33.csv"
pa7="dataset"+"/"+"datanewchi44.csv"


pa8="dataset"+"/"+"essen"+"/"+"MLTRNL.praat"
pa9="dataset"+"/"+"essen"+"/"+"myspsolution.praat"


result_array = np.empty((0, 27))


def mysppron(m,p,q):
	sound=m
	sourcerun=p 
	path=q
	objects= run_file(sourcerun, -20, 2, 0.3, "yes",sound,path, 80, 400, 0.01, capture_output=True)
	#print(objects[0],objects[1]) # This will print the info from the sound object, and objects[0] is a parselmouth.Sound object
	z1=str(objects[1]) # This will print the info from the textgrid object, and objects[1] is a parselmouth.Data object with a TextGrid inside
	z2=z1.strip().split()
	z3=int(z2[13]) # will be the integer number 10
	z4=float(z2[14]) # will be the floating point number 8.3
	db= binom.rvs(n=10,p=z4,size=10000)
	a=np.array(db)
	b=np.mean(a)*100/10
	print ("Pronunciation_posteriori_probability_score_percentage= :%.2f" % (b))
	return round(b,2)
pronunciation = []
files = []
for soundi in os.listdir(audioFilesPath):
	if soundi.endswith('.mp3'):
		files.append(soundi)
		soundi = os.path.join(audioFilesPath,soundi)
		print(soundi)
		#Pronunciation_posteriori_probability_score_percentage
		bi=mysppron(soundi,pa9,audioFilesPath)
		pronunciation.append(bi)
		# feature extraction
		objects= run_file(pa8, -20, 2, 0.3, "yes", soundi, audioFilesPath, 80, 400, 0.01, capture_output=True)
		z1=( objects[1]) # This will print the info from the textgrid object, and objects[1] is a parselmouth.Data object with a TextGrid inside
		z3=z1.strip().split()
		z2=np.array([z3])
		result_array=np.append(result_array,[z3], axis=0)
print(pronunciation)	
np.savetxt(pa1,result_array, fmt='%s',delimiter=',')

#Data and features analysis 
df = pd.read_csv(pa1,
						names = ['avepauseduratin','avelongpause','speakingtot','avenumberofwords','articulationrate','inpro','f1norm','mr','q25',
								'q50','q75','std','fmax','fmin','vowelinx1','vowelinx2','formantmean','formantstd','nuofwrds','npause','ins',
								'fillerratio','xx','xxx','totsco','xxban','speakingrate'],na_values='?')

scoreMLdataset=df.drop(['xxx','xxban'], axis=1)
scoreMLdataset.to_csv(pa7, header=False,index = False)
newMLdataset=df.drop(['avenumberofwords','f1norm','inpro','q25','q75','vowelinx1','nuofwrds','npause','xx','totsco','xxban','speakingrate','fillerratio'], axis=1)
newMLdataset.to_csv(pa5, header=False,index = False)
namess=nms = ['avepauseduratin','avelongpause','speakingtot','articulationrate','mr',
								'q50','std','fmax','fmin','vowelinx2','formantmean','formantstd','ins',
								'xxx']
df1 = pd.read_csv(pa5,
						names = namess)
df33=df1.drop(['xxx'], axis=1)
print(df33)
array = df33.values
array=np.log(array)
x = array[:,0:13]

print(" ")
print(" ")
print("====================================================================================================")
print("HERE ARE THE RESULTS, your spoken language level (speaking skills).")
print("a: just started, a1: beginner, a2: elementary, b1: intermediate, b2: upper intermediate, c: master") 
print("====================================================================================================")

filename="dataset"+"/"+"essen"+"/"+"CART_model.sav"
model = pickle.load(open(filename, 'rb'))
predictions_CART_model = model.predict(x)
print("58% accuracy    ",predictions_CART_model)

filename="dataset"+"/"+"essen"+"/"+"KNN_model.sav"
model = pickle.load(open(filename, 'rb'))
predictions_KNN_model = model.predict(x)
print("65% accuracy    ",predictions_KNN_model)

filename="dataset"+"/"+"essen"+"/"+"LDA_model.sav"
model = pickle.load(open(filename, 'rb'))
predictions_LDA_model = model.predict(x)
print("70% accuracy    ",predictions_LDA_model)

filename="dataset"+"/"+"essen"+"/"+"LR_model.sav"
model = pickle.load(open(filename, 'rb'))
predictions_LR_model = model.predict(x)
print("67% accuracy    ",predictions_LR_model)

filename="dataset"+"/"+"essen"+"/"+"NB_model.sav"
model = pickle.load(open(filename, 'rb'))
predictions_NB_model = model.predict(x)
print("64% accuracy    ",predictions_NB_model)

filename="dataset"+"/"+"essen"+"/"+"SVN_model.sav"
model = pickle.load(open(filename, 'rb'))
predictions_SVN_model = model.predict(x)
print("63% accuracy    ",predictions_SVN_model)

dataSave = pd.DataFrame({
	"files" : files,
	"pronunciation" : pronunciation,
	"predictions_CART_model" : predictions_CART_model,
	"predictions_KNN_model" : predictions_KNN_model,
	"predictions_LDA_model" : predictions_LDA_model,
	"predictions_LR_model" : predictions_LR_model,
	"predictions_NB_model" : predictions_NB_model,
	"predictions_SVN_model" : predictions_SVN_model
})

# Result file name 
dataSave.to_csv('vahan-TC-20-Dataset-Result.csv',index=False)
