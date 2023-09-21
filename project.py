import zipfile
import os
import io
import numpy as np 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import shutil
from sklearn.feature_extraction.text import CountVectorizer

def extractFile(zipFilePath):
    with zipfile.ZipFile(zipFilePath, "r") as zipRef:
        zipRef.extractall("./")
def moveZipsToRoot(folderName):
    for fileName in os.listdir(os.path.join("./",folderName)):
        shutil.move(os.path.join("./"+folderName, fileName), os.path.join("./", fileName))

# extractFile("./project1_datasets.zip")
# moveZipsToRoot("project1_datasets")

# with zipfile.ZipFile("./project1_datasets.zip", "r") as zipRef:
#     print(zipRef.namelist())
spamData = []
hamData = []
with zipfile.ZipFile("./enron1_train.zip", "r") as zipRef:
    spamFiles = [f for f in zipRef.namelist() if f.endswith("spam.txt")]
    hamFiles = [f for f in zipRef.namelist() if f.endswith("ham.txt")]
    for file in spamFiles:
        spamData.append(io.TextIOWrapper(zipRef.open(file), encoding='iso-8859-1', newline='').read())
    for file in hamFiles:
        hamData.append(io.TextIOWrapper(zipRef.open(file), encoding='iso-8859-1', newline='').read())
    
    # for innerFile in zipFiles:
    #     innerFileData = zipRef.read(innerFile)
    #     print(innerFileData)
    #     innerZip = zipfile.ZipFile(innerFileData, 'r')
    #     spamFiles = [f for f in innerZip.namelist() if f.endswith("spam.txt")]
    #     hamFiles = [f for f in innerZip.namelist() if f.endswith("ham.txt")]
    #     for file in spamFiles:
    #         spamData.append(innerZip.read(file, "r"))
    #     for file in hamFiles:
    #         hamData.append(innerZip.read(file, "r"))

print(spamData[0])
print(hamData[0])


class spamClassifier:
    def __init__(self, dataSetName):
        self.dataSetName = dataSetName
        self.trainData = []
        self.testData = []
    def readZipFile():
        zipFiles = [f for f in os.listdir("./") if f.startswith(self.dataSetName)]
        pass
    