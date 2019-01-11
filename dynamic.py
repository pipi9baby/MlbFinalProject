from collections import defaultdict
import json
import time

NCEukFile = r"C:\Users\MINHAN\Desktop\signalp\test\NCEuk.nr.fasta"
SPEukFile = r"C:\Users\MINHAN\Desktop\signalp\test\SPEuk.nr.fasta"
TMEukFile = r"C:\Users\MINHAN\Desktop\signalp\test\TMEuk.nr.fasta"
trainFile = r"C:\Users\MINHAN\Desktop\signalp\train\train.fasta"
outputDir = r"C:\Users\MINHAN\Desktop\structure96"

aminoNum = 96

aminoDict ={'A':0,'V':1,'L':2,'I':3,'F':4,'W':5,'M':6,'P':7,'G':8,'Y':9,'N':10,'Q':11,'S':12,'T':13,'C':14,'D':15,'E':16,'H':17,'K':18,'R':19}
#分子量
#aminoDict ={'G':0,'A':1,'V':2,'L':3,'I':4,'F':5,'W':6,'Y':7,'D':8,'H':9,'N':10,'E':11,'K':12,'Q':13,'M':14,'R':15,'S':16,'T':17,'C':18,'P':19}
#A-z
#aminoDict ={'A':0,'C':1,'D':2,'E':3,'F':4,'G':5,'H':6,'I':7,'K':8,'L':9,'M':10,'N':11,'P':12,'Q':13,'R':14,'S':15,'T':16,'V':17,'W':18,'Y':19}
#動態規劃法，輸入兩個string回傳一個int
"""
def dynamicString(str1, str2):
	test_list=[ [999] * (len(str2)+1) for i in range(len(str1)+1) ]

	#初始化邊界的數值
	for i in range(len(str2)+1):
		test_list[0][i] = i
	for i in range(len(str1)+1):
		test_list[i][0] = i

	#開始做動態規劃
	for i in range(1,len(str1)+1):
		for j in range(1,len(str2)+1):
			#相等
			if(str1[i-1] == str2[j-1]):
				test_list[i][j] = test_list[i-1][j-1]
			else:
				test_list[i][j] = 1 + min([test_list[i-1][j-1], test_list[i-1][j], test_list[i][j-1]])
	return test_list[len(str1)][len(str2)]
"""
def dynamicString(str1, str2):
	count = 0
	for i in range(0,len(str1)):
		if str1[i] == str2[i]:
			count += 1
		else:
			return count
	return count

#讀training data的seq和ans
#string list list
def ReadTrainDataSeq(filePath, trainDataSeq, trainY):
	fr = open(filePath)
	train = fr.read()
	fr.close()
	trainLi = train.split(">")
	for eachData in trainLi:
		if eachData == "":
			continue

		tmpLi = eachData.split("\n")
		trainDataSeq.append(tmpLi[1][:aminoNum])
		if "S" in tmpLi[2]:
			trainY.append(1)
		else:
			trainY.append(0)

#讀tese data的seq和ans
#string list
def ReadTestDataSeq(filePath):
	fr = open(filePath)
	test = fr.read()
	fr.close()
	testLi = test.split(">")
	testDataSeq = []
	for eachData in testLi:
		if eachData == "":
			continue
		tmpLi = eachData.split("\n")
		testDataSeq.append(tmpLi[1][:aminoNum])
	return testDataSeq

#找seq最像的1個
def FindCloset1Seq(seq1, DataSeq):
	simDict = defaultdict(list)
	for seq2 in DataSeq:
		returnInt = dynamicString(seq1, seq2)
		simDict[returnInt].append(seq2)

	#找最大的20個
	items = list(simDict.keys())

	minIndex = min(items)	
	return simDict[minIndex][0]
"""
def FindCloset20Seq(seq1, DataSeq):
	simDict = defaultdict(list)
	seqSimLi = []
	for seq2 in DataSeq:
		returnInt = dynamicString(seq1, seq2)
		simDict[returnInt].append(seq2)

	#找最大的20個
	items = list(simDict.keys())
	items.sort()
	for key in items:
		if(len(seqSimLi) >= 20):
			break
		value = simDict[key]
		for eachSeq in value:
			if(len(seqSimLi) >= 20):
				break
			index = DataSeq.index(eachSeq)
			seqSimLi.append(trainY[index])

	if(len(seqSimLi) < 20):
		for i in range(20 - len(seqSimLi)):
			seqSimLi.append(0)
	elif(len(seqSimLi) > 20):
		print("Error")
		seqSimLi = seqSimLi[:20]
	return seqSimLi
"""

def OneHotEncoding(seq):
	result = []
	for amino in seq:
		#做one hot encoding
		tmpLi = [0] * 20
		try:
			tmpLi[aminoDict[amino]] = 1
		except:
			print(seq1, amino)
		result = result + tmpLi
	if len(seq) < aminoNum:
		tmpLi = [0] * ((aminoNum - len(seq))*20)
		result = result + tmpLi
	return result

localtime = time.asctime( time.localtime(time.time()) )
print("start time", localtime)
print("開始讀取資料")
trainDataSeq = []
trainY = []

ReadTrainDataSeq(trainFile, trainDataSeq, trainY)

#輸出trainY
with open('trainY.json', 'w') as outfile:
	json.dump(trainY, outfile)

testDataSeq = ReadTestDataSeq(NCEukFile)

print("do trainX")
trainX = []
seqSimLi = []
for seq1 in trainDataSeq:
	#取train data最像的1個
	seqSimLi.append(FindCloset1Seq(seq1, trainDataSeq))
	for amino in seq1:
		#做one hot encoding
		tmpLi = [0] * 20
		try:
			tmpLi[aminoDict[amino]] = 1
		except:
			print(seq1, amino)
		trainX = trainX + tmpLi
	if len(seq1) < aminoNum:
		tmpLi = [0] * ((aminoNum - len(seq1))*20)
		trainX = trainX + tmpLi


#輸出trainX
with open(outputDir + r'\trainX.json', 'w') as outfile:
	json.dump(trainX, outfile)

#做最像的seq的encoding
trainXP = []
for seq in seqSimLi:
	trainXP = trainXP + OneHotEncoding(seq)

#輸出trainX的pair seq
with open(outputDir + r'\trainX_pair.json', 'w') as outfile:
	json.dump(trainXP, outfile)

print("do NCEuk")
#test data最像的20個
testX = []
seqSimLi = []
for seq1 in testDataSeq:
	seqSimLi.append(FindCloset1Seq(seq1, testDataSeq))
	for amino in seq1:
		#做one hot encoding
		tmpLi = [0] * 20
		tmpLi[aminoDict[amino]] = 1
		testX = testX + tmpLi
	if len(seq1) < aminoNum:
		tmpLi = [0] * ((aminoNum - len(seq1))*20)
		testX = testX + tmpLi

#輸出NCEuk testData
with open(outputDir + r'\NCEuk.json', 'w') as outfile:
	json.dump(testX, outfile)

#做最像的seq的encoding
testXP = []
for seq in seqSimLi:
	testXP = testXP + OneHotEncoding(seq)
#輸出testX的pair seq
with open(outputDir + r'\NCEuk_pair.json', 'w') as outfile:
	json.dump(testXP, outfile)

print("do SPEuk")
testDataSeq = ReadTestDataSeq(SPEukFile)
#test data最像的20個
testX = []
seqSimLi = []
for seq1 in testDataSeq:
	seqSimLi.append(FindCloset1Seq(seq1, testDataSeq))
	for amino in seq1:
		#做one hot encoding
		tmpLi = [0] * 20
		tmpLi[aminoDict[amino]] = 1
		testX = testX + tmpLi
	if len(seq1) < aminoNum:
		tmpLi = [0] * ((aminoNum - len(seq1))*20)
		testX = testX + tmpLi

with open(outputDir + r'\SPEuk.json', 'w') as outfile:
	json.dump(testX, outfile)

#做最像的seq的encoding
testXP = []
for seq in seqSimLi:
	testXP = testXP + OneHotEncoding(seq)
#輸出testX的pair seq
with open(outputDir + '\SPEuk_pair.json', 'w') as outfile:
	json.dump(testXP, outfile)

print("do TMEuk")
testDataSeq = ReadTestDataSeq(TMEukFile)
#test data最像的20個
testX = []
seqSimLi = []
for seq1 in testDataSeq:
	seqSimLi.append(FindCloset1Seq(seq1, testDataSeq))
	for amino in seq1:
		#做one hot encoding
		tmpLi = [0] * 20
		tmpLi[aminoDict[amino]] = 1
		testX = testX + tmpLi
	if len(seq1) < aminoNum:
		tmpLi = [0] * ((aminoNum - len(seq1))*20)
		testX = testX + tmpLi

with open(outputDir + r'\TMEuk.json', 'w') as outfile:
	json.dump(testX, outfile)

#做最像的seq的encoding
testXP = []
for seq in seqSimLi:
	testXP = testXP + OneHotEncoding(seq)
#輸出testX的pair seq
with open(outputDir + r'\TMEuk_pair.json', 'w') as outfile:
	json.dump(testXP, outfile)

localtime = time.asctime( time.localtime(time.time()) )
print("END time", localtime)

print("successful!")