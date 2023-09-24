import zipfile
import os
import io
import glob
import numpy as np 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import shutil
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, log_loss

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
        self.testResults = None
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
        self.testResults = self._calc(self.data.bagOfWords["test"][:, -1], self.predictions)

    def _calc(self, Y, YPred):
        accuracy = accuracy_score(Y, YPred)
        precision = precision_score(Y, YPred, zero_division=0.0, average="binary")
        recall = recall_score(Y, YPred, average="binary")
        fscore = f1_score(Y, YPred, average="binary")
        temp = [accuracy, precision, recall, fscore]
        return [round(_, 3) for _ in temp]


class bernoulliNB:
    def __init__(self,rep):
        self.priorsProb = {1:None, 0:None}
        self.priorCounts = {1:None, 0:None}
        self.condProb = {1:[], 0:[]}
        self.data = rep
        self.switchedKeyValTrain = {val: key for key, val in self.data.bernoulliIndexedFeatures["train"].items()}
        self.switchedKeyValTest = {val:key for key, val in self.data.bernoulliIndexedFeatures["test"].items()}
        self.predictions = None
        self.testResults = None
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
        self.testResults = self._calc(self.data.bernoulli["test"][:, -1], self.predictions)
    
    def _calc(self, Y, YPred):
        accuracy = accuracy_score(Y, YPred)
        precision = precision_score(Y, YPred, zero_division=0.0, average="binary")
        recall = recall_score(Y, YPred, average="binary")
        fscore = f1_score(Y, YPred, average="binary")
        temp = [accuracy, precision, recall, fscore]
        return [round(_, 3) for _ in temp]

class mcapLR:
    def __init__(self, rep, dataSetPrefix):
        self.data = rep
        # Y1 = np.ones(self.data.bagOfWords["train"].shape[0], dtype=int).reshape(-1, 1)
        # self.trainDataBagOfWords = np.hstack((Y1, self.data.bagOfWords["train"]))
        # self.testDataBagOfWords = self._testDataAlignment(rep.bagOfWords, rep.bagOfWordsIndexedFeatures)
        # Y2 = np.ones(self.data.bernoulli["train"].shape[0], dtype=int).reshape(-1, 1)
        # self.trainDataBernoulli = np.hstack((Y2, self.data.bernoulli["train"]))
        # self.testDataBernoulli = self._testDataAlignment(rep.bernoulli, rep.bernoulliIndexedFeatures)
        self.trainDataBagOfWords, self.testDataBagOfWords = self._dataAlignment(self.data.bagOfWords, self.data.bagOfWordsIndexedFeatures)
        self.bagOfWordParametersTuning = []
        self.trainDataBernoulli, self.testDataBernoulli = self._dataAlignment(self.data.bernoulli, self.data.bernoulliIndexedFeatures)
        self.bernoulliParametersTuning = []
        self.lambdaCandidates = [0.02, 0.06, 0.1, 0.2, 0.4]
        self.iterationsCandidates = [50,150,450,1350]
        self.learningRatesCandidates = [0.001, 0.01, 0.1, 0.5]
        self.defaultLearningRate = 0.1
        self.defaultlambda = 0.02
        self.defaultIterations = 1000
        self._counter = 0
        self._plotNames = ["BagOfWords_ParametersTuning.png", "Bernoulli_ParametersTuning.png"]
        self._dataSetPrefix = dataSetPrefix
    
    def _dataAlignment(self, dataModel, indexedFeatures):
        lookUpdic = {val: key for key, val in indexedFeatures["test"].items()}
        newMatrixTrain = np.ones(dataModel["train"].shape[0], dtype=int).reshape(-1, 1)
        newMatrixTest = np.ones(dataModel["test"].shape[0], dtype=int).reshape(-1, 1)
        for word, index in indexedFeatures["train"].items():
            if word in lookUpdic:
                newMatrixTest = np.hstack((newMatrixTest, dataModel["test"][:, lookUpdic[word]].reshape(-1, 1)))
                newMatrixTrain = np.hstack((newMatrixTrain, dataModel["train"][:, index].reshape(-1, 1)))
        newMatrixTrain = np.hstack((newMatrixTrain, dataModel["train"][:, -1].reshape(-1, 1)))
        newMatrixTest = np.hstack((newMatrixTest, dataModel["test"][:, -1].reshape(-1,1)))
        return newMatrixTrain, newMatrixTest
    def _sigmoid(self, product):
        newArray = np.zeros(len(product), dtype=int)
        for i in range(len(newArray)):
            if product[i] < 0:
                newArray[i] = np.exp(product[i])/(1 + np.exp(product[i]))
            else:
                newArray[i] = 1/(1 + np.exp(-product[i]))
        return newArray
    
    def _splitTrainAndValidation(self, dataSet):
        XTrain, XValidation, YTrain, YValidation = train_test_split(dataSet[:, :-1], dataSet[:, -1], test_size=0.3, stratify=dataSet[:, -1], random_state=80)
        return np.hstack((XTrain, YTrain.reshape(-1, 1))), np.hstack((XValidation, YValidation.reshape(-1, 1)))
    
    def gridSearch(self, dataSet, results):
        for i in self.lambdaCandidates:
            list1 = []
            for j in self.learningRatesCandidates:
                list2 = []
                trainData, validationData = self._splitTrainAndValidation(dataSet)
                for k in self.iterationsCandidates:
                    weights, CLL = self.train(i, j, k, trainData)
                    accuracy, precision, recall, fscore = self.validation(weights, validationData)
                    listInner = [CLL, accuracy, precision, recall, fscore]
                    list2.append(listInner)
                list1.append(list2)
            results.append(list1)
    
    def plotting(self, tuningParameters):
        tempList = [[] for _ in range(5)]
        XAxis = []
        for i in range(len(self.lambdaCandidates)):
            for j in range(len(self.learningRatesCandidates)):
                for k in range(len(self.iterationsCandidates)):
                    XAxis.append(str(i)+str(j)+str(k))
                    for l in range(5):
                        tempList[l].append(tuningParameters[i][j][k][l])
        X = np.arange(len(XAxis))
        plotLabels = ["CLL", "Accuracy", "Precision", "Recall", "F-score"]
        fig, axs = plt.subplots(5, 1, sharex=True, figsize=(14, 12))
        for i, val in enumerate(plotLabels):
            axs[i].plot(X, tempList[i], color="g", label=val)
            axs[i].set_ylabel(plotLabels[i])
        axs[-1].set_xticks(X)
        axs[-1].set_xticklabels(XAxis, rotation=90) 
        
        plt.tight_layout()
        plt.savefig(f"./{self._dataSetPrefix}_{self._plotNames[self._counter]}")
        self._counter += 1
        plt.close()
        
    def train(self, lambdaValue, learningRateValue, iterationValue, trainData):
        weights = np.random.uniform(-0.05, 0.05, trainData.shape[1] - 1)
        for _ in range(iterationValue):
            P = self._sigmoid(np.dot(trainData[:, :-1], weights))
            weights = weights + learningRateValue * np.dot(trainData[:, :-1].T, P) - learningRateValue * lambdaValue * weights
        CLL = np.sum(np.dot(trainData[:, -1].T, np.dot(trainData[:, :-1], weights)) - np.log(1+np.exp(np.dot(trainData[:, :-1], weights)))) - lambdaValue * np.sum(np.square(weights))
        return weights, CLL
    def trainUsingTunedParams(self, lambdaValue, learningRateValue, iterationValue, trainData, testData):
        weights, CLL = self.train(lambdaValue, learningRateValue, iterationValue, trainData)
        return self.validation(weights, testData)
        
    def validation(self, weights, validationData):
        predictions = (np.dot(validationData[:, :-1], weights)>0).astype(int)
        accuracy = accuracy_score(validationData[:, -1], predictions)
        precision, recall, fscore, support = precision_recall_fscore_support(validationData[:, -1], predictions, average="binary")
        temp = [accuracy, precision, recall, fscore]
        return [round(_, 3) for _ in temp]

class SGDSklearn:
    def __init__(self, mcapModel):
        self.paramGrid = {
            "alpha": [0.02, 0.06, 0.1, 0.2, 0.4],
            "max_iter": [50, 1350, 450, 1350]
        }
        self.bestParams = None
        self.trainDataBagOfWords, self.testDataBagOfWords = mcapModel.trainDataBagOfWords, mcapModel.testDataBagOfWords
        self.trainDataBernoulli, self.testDataBernoulli = mcapModel.trainDataBernoulli, mcapModel.testDataBernoulli
        self.scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, zero_division=0.0, average="binary"),
            'recall': make_scorer(recall_score, average="binary"),
            'f1_score': make_scorer(f1_score, average="binary"),
            'log_loss': make_scorer(log_loss, greater_is_better=False)
        }
        self.bagOfWordsBestClassifer = None
        self.bernoulliBestClassifier = None
        self._counter = 0
        self.defaultlambda = 0.02
        self.defaultIterations = 50
        self.bagOfWordsResults = []
        self.bernoulliResults = []
        self.bagOfWordsGridSearch = []
        self.bernoulliGridSearch = []
    
    def _splitTrainAndValidation(self, dataSet):
        XTrain, XValidation, YTrain, YValidation = train_test_split(dataSet[:, :-1], dataSet[:, -1], test_size=0.3, stratify=dataSet[:, -1], random_state=80)
        return np.hstack((XTrain, YTrain.reshape(-1, 1))), np.hstack((XValidation, YValidation.reshape(-1, 1)))
    def search(self, trainData, testData):
        sgdClassifier = SGDClassifier(loss="log_loss", learning_rate="constant", eta0=0.01, random_state=80)
        gridSearch = GridSearchCV(sgdClassifier, self.paramGrid, scoring=self.scoring, refit="log_loss", n_jobs=-1)
        Train, Validation = self._splitTrainAndValidation(trainData)
        gridSearch.fit(Train[:, :-1], Train[:, -1])
        self.bestParams = gridSearch.best_params_
        if self._counter == 0:
            self.bagOfWordsBestClassifer = gridSearch.best_estimator_
            self.bagOfWordsResults = self._calc(testData[:, -1], self.bagOfWordsBestClassifer.predict(testData[:, :-1]))
            self.bagOfWordsGridSearch = self._calc(Validation[:,-1], gridSearch.best_estimator_.predict(Validation[:, :-1]))
        else:
            self.bernoulliBestClassifier = gridSearch.best_estimator_
            self.bernoulliResults = self._calc(testData[:, -1], self.bernoulliBestClassifier.predict(testData[:, :-1]))
            self.bernoulliGridSearch = self._calc(Validation[:,-1], gridSearch.best_estimator_.predict(Validation[:, :-1]))
        self._counter += 1
    def _calc(self, Y, YPred):
        accuracy = accuracy_score(Y, YPred)
        precision = precision_score(Y, YPred, zero_division=0.0, average="binary")
        recall = recall_score(Y, YPred, average="binary")
        fscore = f1_score(Y, YPred, average="binary")
        temp = [accuracy, precision, recall, fscore]
        return [round(_, 3) for _ in temp]

        
        
        
        


def deleteFiles():
    allZipFiles = glob.glob(os.path.join("./", "enron*")) + glob.glob(os.path.join("./", "*sets"))
    # folder = glob.glob(os.path.join("./", "project1*"))
    for aFile in allZipFiles:
        if aFile.endswith("datasets"):
            os.rmdir(aFile)
        else:
            os.remove(aFile)
    
def main():
    extracted = dataExtracter("project1_datasets")
    extracted.readData()
    deleteFiles()
    for dataSetPrefix in ["enron1", "enron2", "enron4"]:
        print(f"For the dataSet with the prefix {dataSetPrefix}:")
        dataSet = extracted.allData[dataSetPrefix]
        dataSetRep = modelCreator()
        dataSetRep.createRepresentations(dataSet)
        multiNB = multiNomialNB(dataSetRep)
        multiNB.train()
        multiNB.test()
        print(f"multinomial NB results: {multiNB.testResults}")
        
        berNB = bernoulliNB(dataSetRep)
        berNB.train()
        berNB.test()
        print(f"Bernoulli NB results: {berNB.testResults}")

        LR = mcapLR(dataSet1Rep, dataSetPrefix)
        LR.gridSearch(LR.trainDataBagOfWords, LR.bagOfWordParametersTuning)
        LR.gridSearch(LR.trainDataBernoulli, LR.bernoulliParametersTuning)
        LR.plotting(LR.bagOfWordParametersTuning)
        LR.plotting(LR.bernoulliParametersTuning)
        SGD = SGDSklearn(LR)
        SGD.search(SGD.trainDataBagOfWords, SGD.testDataBagOfWords)
        print("Results using SGD classifier for the bag of words representation:")
        print(f"train result:{SGD.bagOfWordsGridSearch}")
        print(f"test: {SGD.bagOfWordsResults}")
        print(f"best params: {SGD.bestParams}")
        SGD.search(SGD.trainDataBernoulli, SGD.testDataBernoulli)
        print("Results using SGD classifier for the Bernoulli representation:")
        print(f"train result:{SGD.bernoulliGridSearch}")
        print(f"test: {SGD.bernoulliResults}")
        print(f"best params: {SGD.bestParams}")

