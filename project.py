import zipfile
import os
import io
import numpy as np 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import shutil
from sklearn.feature_extraction.text import CountVectorizer

class dataExtracter:
    def __init__(self, zipName):
        self.allData = {"spam": [], "ham": []}
        self.spamData = []
        self.hamData = []
        self.folderName = zipName.lower()
        self.zipPath = os.path.join("./", self.folderName+".zip")
    def extractFile():
        with zipfile.ZipFile(self.zipPath, "r") as folder:
            folder.extractall("./")
    def moveZipToRoot():
        for fileName in os.listdir(os.path.join("./",self.folderName)):
            shutil.move(os.path.join("./" + self.folderName, fileName), os.path.join("./", fileName))
    def readData(dataSetName, trainOrTest, spamOrHam):
        with zipfile.ZipFile(os.path.join("./", dataSetName.lower()+"_"+trainOrTest.lower()+".zip"), "r") as zipRef:
            fils = [f for f in zipRef.namelist() if f.endswith(spamOrHam.lower()+".txt")]
            for file in files:
                self.allData[spamOrHam.lower()].append(io.TextIOWrapper(zipRef.open(file), encoding='iso-8859-1', newline='').read())
        
# extractFile("./project1_datasets.zip")
# moveZipsToRoot("project1_datasets")

# with zipfile.ZipFile("./project1_datasets.zip", "r") as zipRef:
#     print(zipRef.namelist())
# spamData = []
# hamData = []
# with zipfile.ZipFile("./enron1_train.zip", "r") as zipRef:
#     spamFiles = [f for f in zipRef.namelist() if f.endswith("spam.txt")]
#     hamFiles = [f for f in zipRef.namelist() if f.endswith("ham.txt")]
#     for file in spamFiles:
#         spamData.append(io.TextIOWrapper(zipRef.open(file), encoding='iso-8859-1', newline='').read())
#     for file in hamFiles:
#         hamData.append(io.TextIOWrapper(zipRef.open(file), encoding='iso-8859-1', newline='').read())
    
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



class spamClassifier:
    def __init__(self, dataSetName):
        self.dataSetName = dataSetName
        self.trainData = []
        self.testData = []
    def readZipFile():
        zipFiles = [f for f in os.listdir("./") if f.startswith(self.dataSetName)]
        pass
    