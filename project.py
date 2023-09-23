import zipfile
import os
import io
import glob
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

class modelCreator:
    def __init__(self):
        # create the bag of word representation. 
        self.bagOfWords = {"train": None, "test":None}
        self.bagOfWordsFeatures = {"train": None, "test": None}
        self.bagOfWordsIndexedFeatures = {"train": None, "test":None}
        # create the bernoulli representation
        self.bernoulli = {"train":None, "test":None}
        self.bernoulliFeatures = {"train":None, "test":None}
        self.bernoulliIndexedFeatures = {"train": None, "test":None}
    def _createVocab(self, rawFeatures, trainOrTest):
        # create the indexed words. Notice that in the train dictionary, the key is the word and value is the index but in the dictionary for test data, it's the opposite. This is for the convenience of lookup
        if trainOrTest == "train":
            newDic = {word:i for i, word in enumerate(rawFeatures[trainOrTest])}
            return newDic
        if trainOrTest == "test":
            newDic = {i:word for i, word in enumerate(rawFeatures[trainOrTest])}
            return newDic
    def _tokenizeText(self,text):
        # tokenize the email
        tokens = word_tokenize(text)
        tokens = [word.lower() for word in tokens if word.isalnum()]
        stopWords = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stopWords]
        return tokens
    def _processData(self, rawText):
        # create a column of y values. For spam emails their values are 1.
        YSpam = np.ones(len(rawText["spam"]), dtype=int)
        YHam = np.zeros(len(rawText["ham"]), dtype=int)
        catList = rawText["spam"] + rawText["ham"]
        tokenizedData = [self._tokenizeText(email) for email in catList]
        Y = np.concatenate((YSpam, YHam))
        return [tokenizedData, Y]
    def _bagOfWords(self, tokenizedData, Y, trainOrTest):
        # create the bag of words representation
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform([' '.join(email) for email in tokenizedData])
        X_dense = X.toarray()
        Y_reshaped = Y.reshape(-1, 1)
        self.bagOfWords[trainOrTest] = np.concatenate((X_dense, Y_reshaped), axis=1)
        self.bagOfWordsFeatures[trainOrTest] = vectorizer.get_feature_names_out()
        self.bagOfWordsIndexedFeatures[trainOrTest] = self._createVocab(self.bagOfWordsFeatures, trainOrTest)
    def _bernoulli(self, tokenizedData, Y, trainOrTest):
        # create the bernoulli representation
        vectorizer = CountVectorizer(binary=True)
        X = vectorizer.fit_transform([' '.join(email) for email in tokenizedData])
        X_dense = X.toarray()
        Y_reshaped = Y.reshape(-1, 1)
        self.bernoulli[trainOrTest] = np.concatenate((X_dense, Y_reshaped), axis=1)
        self.bernoulliFeatures[trainOrTest] = vectorizer.get_feature_names_out()
        self.bernoulliIndexedFeatures[trainOrTest] = self._createVocab(self.bernoulliFeatures, trainOrTest)
    def createRepresentations(self, rawText):
        # invoke the methods for creating the representations
        tokenizedData = None
        Y = None
        for trainOrTest in ["train", "test"]:
            [tokenizedData, Y] = self._processData(rawText[trainOrTest])
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
        classes, counts = np.unique(self.data.bagOfWords["train"][:,-1], return_counts = True)
        for OneClass, count in zip(classes, counts):
            self.priorCounts[OneClass] = count
            self.priorsProb[OneClass] = np.log(count/len(self.data.bagOfWords["train"][:,-1]))
        lengthOfVoca = len(self.data.bagOfWordsFeatures["train"])
        for i in range(lengthOfVoca):
            self.condProb[1].append((np.sum(self.data.bagOfWords["train"][:self.priorCounts[1], i]) + 1) /np.sum(self.data.bagOfWords["train"][:self.priorCounts[1], :-1]))
            self.condProb[0].append((np.sum(self.data.bagOfWords["train"][self.priorCounts[1]:, i]) + 1) /np.sum(self.data.bagOfWords["train"][self.priorCounts[1]:, :-1]))
    def test(self):
        predictions = []
        # ignore the unseen words in the test data
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
        self.predictions = np.array(predictions)


class bernoulliNB:
    def __init__(self,rep):
        self.priorsProb = {1:None, 0:None}
        self.priorCounts = {1:None, 0:None}
        self.condProb = {1:[], 0:[]}
        self.data = rep
        self.switchedKeyValTrain = {val: key for key, val in self.data.bernoulliIndexedFeatures["train"].items()}
        self.switchedKeyValTest = {val:key for key, val in self.data.bernoulliIndexedFeatures["test"].items()}
        self.predictions = None
    def train(self):
        classes, counts = np.unique(self.data.bernoulli["train"][:,-1], return_counts = True)
        for OneClass, count in zip(classes, counts):
            self.priorCounts[OneClass] = count
            self.priorsProb[OneClass] = np.log(count/len(self.data.bernoulli["train"][:,-1]))
        lengthOfVoca = len(self.data.bernoulliFeatures["train"])
        for i in range(lengthOfVoca):
            self.condProb[1].append((np.sum(self.data.bernoulli["train"][:self.priorCounts[1], i]) + 1) / (self.priorCounts[1] + 2))
            self.condProb[0].append((np.sum(self.data.bernoulli["train"][self.priorCounts[1]:, i]) + 1) / (self.priorCounts[0] + 2))
    
    def test(self):
        predictions = []
        # ignore the unseen words in the test data
        # priors = {1: self.priorsProb[1], 0: self.priorsProb[0]}
        # for i, word in self.rep.bagOfWordsIndexedFeatures["test"].items():
        #     if word in self.rep.bagOfWordsIndexedFeatures["train"]:
        for i in range(self.data.bernoulli["test"].shape[0]):
            maxProb = -np.inf
            predicted = None
            for oneClass, prior in self.priorsProb.items():
                prob = prior
                for j, word in self.switchedKeyValTrain.items():
                    if word in self.switchedKeyValTest:
                        if self.data.bernoulli["test"][i,self.switchedKeyValTest[word]]:
                            prob += np.log(self.condProb[oneClass][j])
                    else:
                        prob += np.log(1-self.condProb[oneClass][j])
                if prob > maxProb:
                    maxProb = prob
                    predicted = oneClass
            predictions.append(predicted)
        self.predictions = np.array(predictions)
    
 
def testFunction():
    def accuracy(dataSet, mOrb):
        count = 0
        if mOrb == "m":
            for i in range(len(dataSet.predictions)):
                if dataSet.predictions[i] == dataSet.data.bagOfWords["test"][i,-1]:
                    count += 1
        else:
            for i in range(len(dataSet.predictions)):
                if dataSet.predictions[i] == dataSet.data.bernoulli["test"][i,-1]:
                    count += 1
            
        print(count/len(dataSet.predictions))
    extracted = dataExtracter("project1_datasets")
    extracted.readData()
    allZipFiles = glob.glob(os.path.join("./", "enron*")) + glob.glob(os.path.join("./", "*sets"))
    # folder = glob.glob(os.path.join("./", "project1*"))
    for aFile in allZipFiles:
        if aFile.endswith("datasets"):
            os.rmdir(aFile)
        else:
            os.remove(aFile)
    dataSet1 = extracted.allData["enron1"]
    dataSet1Rep = modelCreator()
    dataSet1Rep.createRepresentations(dataSet1)
    multiNB1 = multiNomialNB(dataSet1Rep)
    multiNB1.train()
    multiNB1.test()
    berNB1 = bernoulliNB(dataSet1Rep)
    berNB1.train()
    berNB1.test()
    accuracy(multiNB1, "m")
    accuracy(berNB1, "b")
    # print(1 - abs(multiNB1.data.bagOfWords["test"][:, -1] - multiNB1.predictions)/len(multiNB1.predictions))

            
    # print(len(dataSet1Rep.bagOfWords["train"][:, 0]))
    # print((len(dataSet1["train"]["spam"])+len(dataSet1["train"]["ham"])))
    # print(len(set(dataSet1Rep.bagOfWordsFeatures["train"])))
    # print(len(dataSet1Rep.bagOfWordsIndexedFeatures["train"]))
    # print(len(set(dataSet1Rep.bagOfWordsFeatures["test"])))
    # print(len(dataSet1Rep.bagOfWordsIndexedFeatures["test"]))

    # remove the zip files and empty folder

    
testFunction()