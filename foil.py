import sys
import math
import string
import numpy as np
import copy


# based on https://cgi.csc.liv.ac.uk/~frans/KDD/Software/FOIL_PRM_CPAR/foil.html

POSVAL = 0 
NEGVAL = 2
MIN_BEST_GAIN = 0.00
MAX_NUM_IN_ANT = 5
K = 5 # best K rules for each class 

smallData   =   np.array([[0,2],
					       [1,2],
					       [0,3],
					       [1,3]])

smallLabel = np.array([0,0,0,2])




pimaData =  np.array([[2,9,13,17,21,28,32,38], 
			 			[1,8,13,17,21,27,31,36], 
			 			[3,10,13,16,21,27,32,36], 
						[1,8,13,17,21,28,31,36], 
						[1,9,12,17,21,29,35,37], 
						[2,8,14,16,21,27,31,36], 
						[1,7,13,17,21,28,31,36], 
						[3,8,11,16,21,28,31,36], 
						[1,10,13,18,24,28,31,38], 
						[3,9,14,16,21,26,31,38], 
						[2,8,14,16,21,28,31,36], 
						[3,10,14,16,21,28,31,37],
						[3,9,14,16,21,28,33,39], 
						[1,10,13,17,25,28,31,39], 
						[2,10,13,16,22,27,32,38]])


pimaLabel = np.array([42,41,42,41,42,41,42,41,42,42,41,42,41,42,42])



dataBinary = np.array([
#	 0 1 2 3 4 5 6 7 8 9 10111213141516
	[0,0,1,0,1,1,1,1,1,1,1,0,0,1,1,1,1], #3
	[0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0], #1
	[0,0,0,0,1,0,1,1,1,1,1,0,0,0,0,0,0], #1
	[0,0,1,0,1,0,1,1,1,1,1,0,0,0,0,0,0], #1
	[1,0,1,0,1,0,1,1,1,1,1,0,0,0,0,0,1], #2
	[0,0,1,0,1,1,1,1,1,1,1,0,0,0,0,1,0], #2
	[1,0,1,0,1,0,1,1,1,1,1,0,0,0,0,1,0], #2
	])

labelNonBin = np.array([3,1,1,1,2,2,2]);

pimaPath = "pima.tsv"



def readFile(tsv):
    file = open(tsv,"r")
    content = file.readlines()
    lst = []
    for i in range(0, len(content)):
        row = (content[i][:-1]).split(" ")
        num = [int(i) for i in row]
       # print(row)
        lst.append(num)
    return(np.array(lst))


def getFeature(d):
	return d[:,:-1]


def getLabel(d):
	return d[:, -1]


def predAccu(tru, pred):
	match = np.sum(tru == pred) 
	total = len(tru)
	accu = (float(match) / float(total))
	return accu



def sortTup(tup):
	tupSorted = sorted(tup, key=lambda x: x[1], reverse=True)
	return tupSorted



def nonBinary(a):
	for row in range(np.shape(a)[0]):
		for col in range(1, np.shape(a)[1]):
			a[row,col] = a[row,col] + col*2
	return a



def getPosOrNegExamples(feature, lab, pos):
	posInd, negInd = np.where(lab == pos)[0], np.where(lab != pos)[0]
	posEx, negEx = feature[posInd, :], feature[negInd, :]
	return [posEx, negEx]


# check whether any value in set (of antecedents) is in array
# s is a set/list, a is 1-D array
def checkSubset(a, s):
	if not s: return True 
	#if not a: return False  
	for i in s:
		if i in a: return True
	return False


def rankPair(rules, accu):  
    rankedR = [r for r, _ in sorted(zip(rules, accu), reverse = True)]
    rankedA = sorted(accu, reverse = True)
    return [rankedR, rankedA]


def uniqueAttributes(a):
	return np.unique(a)


# P' and N' adjusted so that all examples which do not contain attribute are removed
def retainExamples(attr, ex):
	retain = np.apply_along_axis(checkSubset, 1, ex, attr)
	retainEx = ex[retain, :]
	return retainEx


# remove all examples from P (NOT P') that satisfy the rule
def removeExamples(attr, ex):
	remove = np.invert(np.apply_along_axis(checkSubset, 1, ex, attr))
	removeEx = ex[remove, :]
	return removeEx


def gain(p, p2, n, n2):
	# Calculate gain
	sP, sP2 = np.shape(p)[0], np.shape(p2)[0]
	sN, sN2 = np.shape(n)[0], np.shape(n2)[0]
	# gain(a) = |P'| (log(|P'|/|P'|+|N'|) - log(|P|/|P|+|N|))

	if sP2 == 0: return 0 
	gain = sP2 * (math.log(float(sP2) / float(sP2 + sN2)) - math.log(float(sP) / float(sP + sN))) 
	return gain


def calculateGain(ant, pos, neg):
	pos2 = retainExamples(ant, pos)
	neg2 = retainExamples(ant, neg)
	g = gain(pos, pos2, neg, neg2)
	return g




class Classification(object):
	def __init__(self, feature, label):
		self.feature = feature
		self.label = label 
		self.uniqueAttr = np.unique(feature)
		self.uniqueLab = np.unique(label)
		self.numAttribute = self.uniqueAttr.size
		self.numClass = np.unique(self.label).size
		self.attrArray = np.zeros((2,self.numAttribute))
		self.rules = [] # Rules: [({ante, ante},con), ({ante},con) ...]
		self.con = None
		self.ruleAccu = []
		

		# Those get changed once per outerloop
		self.posExamples = None
		self.negExamples = None


		# Those get changed inside the innerloop
		self.posExamples2 = None
		self.negExamples2 = None
		self.attrArray2 = None
		self.ant = set()
		
		

	# get positive and negative examples, define positive
	def setPosNeg(self, feature, lab, posVal):
		[pos, neg] = getPosOrNegExamples(feature, lab, posVal)
		self.posExamples, self.negExamples = pos, neg
		# learn towards pos
		self.con = posVal


	# make copies and set ante to empty set for inner loop
	def reset(self):
		self.ant = set()
		self.posExamples2 = copy.deepcopy(self.posExamples)
		self.negExamples2 = copy.deepcopy(self.negExamples)
		self.attrArray2 = copy.deepcopy(self.attrArray)


	# does not modify attrArray2 
	def calculateGains(self):

		attrCopy = copy.deepcopy(self.attrArray2)
		calculated = attrCopy[0]
		attrCopy[1] = np.zeros(len(attrCopy[1]))

		for i in range(len(calculated)):
			if not calculated[i]: 
				tempAnt = copy.deepcopy(self.ant)
				tempAnt.add(self.uniqueAttr[i])
				# If new ant were to be added 
			
				attrCopy[1][i] = calculateGain(tempAnt, self.posExamples2, self.negExamples2)
		return attrCopy



	def noGain(self):
		attrGains = self.calculateGains()[1]
		if np.any(attrGains > MIN_BEST_GAIN): return False
		return True




	# for each rule, calculate accuracy = (Nc+1)/(Ntot+numberOfClasses)
	def getAccuracy(self):
		ind = np.apply_along_axis(checkSubset, 1, self.feature, self.ant)
		total = np.sum(ind)
		lab = self.label[ind]
		count = np.sum(lab == self.con)
		accu = (float(count) + 1) / (float(total) + float(self.numClass))
		return accu


	# return a list rules
	def foilGeneration(self, rule, con):
	#	if newRule == None: newRule = tuple()

		self.attrArray2 = self.calculateGains()

		maxInd = np.argmax(self.attrArray2[1]) # indices of the FIRST occurrence, what if same gain ???

		bestGain = self.attrArray2[1][maxInd]
		bestAttr = self.uniqueAttr[maxInd]
		#print("bestGain", bestGain)

		if bestGain <= MIN_BEST_GAIN: 
			#print("< MIN BEST GAIN")
			# Found a rule
			rule.append((self.ant, con))
			return rule
		
		self.ant.add(bestAttr) # union 
		#print("union", self.ant)
		self.attrArray2[0][maxInd] = 1
		
			
		self.posExamples2 = retainExamples(self.ant, self.posExamples2)
		self.negExamples2 = retainExamples(self.ant, self.negExamples2)
		#print("pos2", self.posExamples2)
		#print("neg2", self.negExamples2)


		if len(self.negExamples2) == 0 or len(self.ant) > MAX_NUM_IN_ANT:

			rule.append((self.ant, con))
			#print("attr array", self.attrArray2)
			return rule

		self.foilGeneration(rule, con)






	def startFOIL(self):

		for labVal in self.uniqueLab:
			self.setPosNeg(self.feature, self.label, labVal)		

			while len(self.posExamples) != 0:

				self.reset()

				# if no attributes exist that can produce a gain above minimum break
				if self.noGain(): break

				newRule = []
				self.foilGeneration(newRule, self.con)

				newAnt = newRule[0][0]
				accu = self.getAccuracy()
				#print("rule and its accu:", newRule, str(accu))

				#print("new rule found", newRule)
				#print("new ant", newAnt)

				# After one inner loop, P gets changed once
				self.rules.append(newRule[0])
				self.ruleAccu.append(accu)

				self.posExamples = removeExamples(newAnt, self.posExamples)
				




	def predict(self, newdata, k):
		# Obtain all rules whose antecedent is a subset of the given record
		candidateRules, candidateAccu = [], []
		for r in range(len(self.rules)):
			if checkSubset(newdata, self.rules[r][0]):
				candidateRules.append(self.rules[r])
				candidateAccu.append(self.ruleAccu[r])

		#print("candidateRules", candidateRules)		
		#print("candidateAccu", candidateAccu)	

		# select the best K rules for each class according to their Laplace accuarcy
		res = []
		for labVal in self.uniqueLab:
			matchRule, matchAccu = [], []
			for i in range(len(candidateRules)):
				if candidateRules[i][1] == labVal: 
					matchRule.append(candidateRules[i])
					matchAccu.append(candidateAccu[i])
			
			# Rank them
			[rankedRules, rankedAccu] = rankPair(matchRule, matchAccu)
			#print("select",[rankedRules, rankedAccu] )

			meanAccu = 0 if not rankedRules else sum(rankedAccu[:k]) / len(rankedAccu[:k])
			res.append((labVal, meanAccu))

		sortedTup = sortTup(res)
		#print(res)	

		# Determine the average expected accuracy over the selected rules for each class
		#print("sorted", sortedTup)

		predLabel = sortedTup[0][0]  # What if multiple labels have same accuracy????
		return predLabel


	def predictAll(self, newdata, k):
		#predLab = np.apply_along_axis(self.predict(), 1, self = self, newdata = newdata, k = k)
		predRes = np.array([])
		for row in newdata:
			predLab = self.predict(row, k)
			predRes = np.append(predRes, predLab)
		return predRes



	




def startClassification(trainData, trainLabel, testData, testLabel, k):
	foilObj = Classification(trainData, trainLabel)
	foilObj.startFOIL()
	pred = foilObj.predictAll(testData, k)
	accuracy = predAccu(testLabel, pred)
	print("Rules generated:", foilObj.rules)
	print("Predicted labels:", pred)
	print("Accuracy on training:", accuracy)





if(__name__ == "__main__"):

	# Test on training data: 

	# small Data 
	print("Small Data")
	startClassification(smallData, smallLabel, smallData, smallLabel, K)
	print("\n")


	# Non Binary Data
	print("Non Binary Data")
	dataNonBin = nonBinary(dataBinary)
	startClassification(dataBinary, labelNonBin, dataBinary, labelNonBin, K)
	print("\n")

	# Pima 
	print("Small Pima")
	startClassification(pimaData, pimaLabel, pimaData, pimaLabel, K)
	print("\n")

	# Large Pima
	print("Large Pima")
	dt = readFile(pimaPath)
	df = getFeature(dt)
	dl = getLabel(dt)
	startClassification(df, dl, df, dl, K)
	print("\n")

	












