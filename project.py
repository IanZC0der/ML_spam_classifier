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
        self.folderName = zipName.lower()
        self.zipPath = os.path.join("./", self.folderName+".zip")
        self._extractFile()
        self._moveZipToRoot()
    def _extractFile(self):
        with zipfile.ZipFile(self.zipPath, "r") as folder:
            folder.extractall("./")
    def _moveZipToRoot(self):
        for fileName in os.listdir(os.path.join("./",self.folderName)):
            shutil.move(os.path.join("./" + self.folderName, fileName), os.path.join("./", fileName))
    def readData(self,dataSetName, trainOrTest, spamOrHam):
        with zipfile.ZipFile(os.path.join("./", dataSetName.lower()+"_"+trainOrTest.lower()+".zip"), "r") as zipRef:
            files = [f for f in zipRef.namelist() if f.endswith(spamOrHam.lower()+".txt")]
            for file in files:
                self.allData[spamOrHam.lower()].append(io.TextIOWrapper(zipRef.open(file), encoding='iso-8859-1', newline='').read())

extracted = dataExtracter("project1_datasets")
extracted.readData("enron1", "train", "spam")
extracted.readData("enron1", "train", "ham")
print(extracted.allData["spam"][0])
print(extracted.allData["ham"][0])

class spamClassifier:
    def __init__(self):
        self.bagOfWords = {"train": None, "test":None}
        self.bagOfWordsFeatures = {"train": None, "test": None}
        self.bernoulli = {"train":None, "test":None}
        self.bernoulliFeatures = {"train":None, "test":None}
        # self.dataSetName = dataSetName
        # self.trainData = []
        # self.testData = []
    def _tokenizeText(self,text):
        tokens = word_tokenize(text)
        # self.words = None
        tokens = [word.lower() for word in tokens if word.isalnum()]
        stopWords = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stopWords]
        return tokens
    def _processData(self, rawText):
        YSpam = np.ones(len(rawText["spam"]))
        YHam = np.ones(len(rawText["ham"]))
        catList = rawText["spam"] + rawText["ham"]
        tokenizedData = [self._tokenizeText(email) for email in catList]
        return tokenizedData, np.concatenate(YSpam, YHam).T
    def _bagOfWords(self, tokenizedData, Y, trainOrTest):
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform([' '.join(email) for email in tokenizedData])
        self.bagOfWords[trainOrTest] = np.hstack((X, Y))
        self.bagOfWordsFeatures[trainOrTest] = vectorizer.get_feature_names_out()
    def _bernoulli(self, tokenizedData, Y, trainOrTest):
        vectorizer = CountVectorizer(binary=True)
        X = vectorizer.fit_transform([' '.join(email) for email in tokenizedData])
        self.bernoulli[trainOrTest] = np.hstack((X, Y))
        self.bernoulliFeatures[trainOrTest] = vectorizer.get_feature_names_out()
    def createRepresentations(self, rawText, trainOrTest):
        tokenizedData, Y = self._processData(rawText)
        self._bagOfWords(tokenizedData, Y, trainOrTest)
        self.bernoulli(tokenizedData, Y, trainOrTest)
        

        