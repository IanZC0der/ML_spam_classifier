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
        self.allData = {"enron1": {"train": {"spam": [], "ham": []}, "test": {"spam": [], "ham": []}},
                        "enron2": {"train": {"spam": [], "ham": []}, "test": {"spam": [], "ham": []}},
                        "enron4": {"train": {"spam": [], "ham": []}, "test": {"spam": [], "ham": []}}}
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
    def readData(self):
        for dataSetName in ["enron1", "enron2", "enron4"]:
            for trainOrTest in ["train", "test"]:
                with zipfile.ZipFile(os.path.join("./", dataSetName.lower()+"_"+trainOrTest.lower()+".zip"), "r") as zipRef:
                    for spamOrHam in ["spam", "ham"]:
                        files = [f for f in zipRef.namelist() if f.endswith(spamOrHam.lower()+".txt")]
                        for file in files:
                            self.allData[dataSetName][trainOrTest][spamOrHam.lower()].append(io.TextIOWrapper(zipRef.open(file), encoding='iso-8859-1', newline='').read())

# extracted = dataExtracter("project1_datasets")
# extracted.readData()
class modelCreator:
    def __init__(self):
        self.bagOfWords = {"train": None, "test":None}
        self.bagOfWordsFeatures = {"train": None, "test": None}
        self.bagOfWordsIndexedFeatures = {"train": None, "test":None}
        self.bernoulli = {"train":None, "test":None}
        self.bernoulliFeatures = {"train":None, "test":None}
        self.bernoulliIndexedFeatures = {"train": None, "test":None}
    def _createVocab(self, rawFeatures, trainOrTest):
        if trainOrTest == "train":
            newDic = {word:i for i, word in enumerate(rawFeatures[trainOrTest])}
            return newDic
        if trainOrTest == "test":
            newDic = {i:word for i, word in enumerate(rawFeatures[trainOrTest])}
            return newDic
    def _tokenizeText(self,text):
        tokens = word_tokenize(text)
        tokens = [word.lower() for word in tokens if word.isalnum()]
        stopWords = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stopWords]
        return tokens
    def _processData(self, rawText):
        YSpam = np.ones(len(rawText["spam"]))
        YHam = np.zeros(len(rawText["ham"]))
        catList = rawText["spam"] + rawText["ham"]
        tokenizedData = [self._tokenizeText(email) for email in catList]
        return tokenizedData, np.concatenate(YSpam, YHam).T
    def _bagOfWords(self, tokenizedData, Y, trainOrTest):
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform([' '.join(email) for email in tokenizedData])
        self.bagOfWords[trainOrTest] = np.hstack((X, Y))
        self.bagOfWordsFeatures[trainOrTest] = vectorizer.get_feature_names_out()
        self.bagOfWordsIndexedFeatures[trainOrTest] = self._createVocab(self.bagOfWordsFeatures, trainOrTest)
    def _bernoulli(self, tokenizedData, Y, trainOrTest):
        vectorizer = CountVectorizer(binary=True)
        X = vectorizer.fit_transform([' '.join(email) for email in tokenizedData])
        self.bernoulli[trainOrTest] = np.hstack((X, Y))
        self.bernoulliFeatures[trainOrTest] = vectorizer.get_feature_names_out()
        self.bernoulliIndexedFeatures[trainOrTest] = self._createVocab(self.bernoulliFeatures, trainOrTest)
    def createRepresentations(self, rawText, trainOrTest):
        for trainOrTest in ["train", "test"]:
            tokenizedData, Y = self._processData(rawText[trainOrTest])
            self._bagOfWords(tokenizedData, Y, trainOrTest)
            self._bernoulli(tokenizedData, Y, trainOrTest)
        

# multinomial NB model with add-one smoothing
class multiNomialNB:
    def __init__(self, rep):
        self.priorsProb = {1:None, 0:None}
        self.priorCounts = {1:None, 0:None}
        self.condProb = {1:[], 0:[]}
        self.data = rep
        self.predictions = None
    def train(self):
        classes, counts = np.unique(self.data.bagOfWords["train"][:,-1])
        for OneClass, count in zip(classes, counts):
            self.priorCounts[OneClass] = count
            self.priorsProb[OneClass] = np.log(count/len(self.data.bagOfWords["train"][:,-1]))
        lengthOfVoca = len(self.data.bagOfWordsFeatures["train"])
        for i in range(lengthOfVoca):
            self.condProb[1].append((np.sum(self.data.bagOfWords["train"][:self.priorCounts[1], i]) + 1) /np.sum(self.data.bagOfWords["train"][:self.priorCounts[1], :-1]))
            self.condProb[0].append((np.sum(self.data.bagOfWords["train"][self.priorCounts[1]:, i]) + 1) /np.sum(self.data.bagOfWords["train"][self.priorCounts[1]:, :-1]))
    def test(self):
        predictons = []
        # ignore the unseen words in the test data
        # priors = {1: self.priorsProb[1], 0: self.priorsProb[0]}
        # for i, word in self.rep.bagOfWordsIndexedFeatures["test"].items():
        #     if word in self.rep.bagOfWordsIndexedFeatures["train"]:
        for i in range(self.data.bagOfWords["test"].shape[0]):
            maxProb = -np.inf
            predicted = None
            for oneClass, prior in self.priorsProb.items():
                prob = prior
                for j in range(self.data.bagOfWords["test"].shape[1] - 1):
                    if self.data.bagOfWords["test"][i,j] != 0 and self.data.bagOfWordsIndexedFeatures["test"][j] in self.data.bagOfWordsIndexedFeatures["train"]:
                        prob += np.log(self.condProb[oneClass][self.data.bagOfWordsIndexedFeatures["train"][self.data.bagOfWordsIndexedFeatures["test"][j]]])
                if prob > maxProb:
                    maxProb = prob
                    predicted = oneClass
            predictions.append(predicted)
        self.predictions = np.array(predictons).T