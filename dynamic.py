from collections import defaultdict
import json
import time

testFile = r"C:\Users\MINHAN\Desktop\signalp\test\NCEuk.nr.fasta"
trainFile = r"C:\Users\MINHAN\Desktop\signalp\train\train.fasta"

aminoNum = 35
aminoDict ={'G':0,'A':1,'V':2,'L':3,'I':4,'F':5,'W':6,'Y':7,'D':8,'H':9,'N':10,'E':11,'K':12,'Q':13,'M':14,'R':15,'S':16,'T':17,'C':18,'P':19}

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

#找seq最像的20個
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

localtime = time.asctime( time.localtime(time.time()) )
print("start time", localtime)
print("開始讀取資料")
trainDataSeq = []
trainY = []

ReadTrainDataSeq(trainFile, trainDataSeq, trainY)
"""
#輸出trainY
with open('trainY.json', 'w') as outfile:  
	json.dump(trainY, outfile)
"""
testDataSeq = ReadTestDataSeq(testFile)

print("取train data最像的20個")
trainX = []
for seq1 in trainDataSeq:
	#取train data最像的20個
	seqSimLi = []
	seqSimLi = FindCloset20Seq(seq1, trainDataSeq)
	for amino in seq1:
		#做one hot encoding
		tmpLi = [0] * 20
		try:
			tmpLi[aminoDict[amino]] = 1
		except:
			print(seq1, amino)
		trainX = trainX + tmpLi
	#存最後20個像的
	trainX = trainX + seqSimLi

#輸出trainX
with open('trainX.json', 'w') as outfile:  
	json.dump(trainX, outfile)

#test data最像的20個
testX = []
for seq1 in testDataSeq:
	seqSimLi = []
	seqSimLi = FindCloset20Seq(seq1, testDataSeq)
	for amino in seq1:
		#做one hot encoding
		tmpLi = [0] * 20
		tmpLi[aminoDict[amino]] = 1
		testX = testX + tmpLi
	#存最後20個像的
	testX = testX + seqSimLi

#輸出NCEuk testData
with open('NCEuk.json', 'w') as outfile:  
	json.dump(testX, outfile)

testFile = r"C:\Users\MINHAN\Desktop\signalp\test\SPEuk.nr.fasta"
testDataSeq = ReadTestDataSeq(testFile)
#test data最像的20個
testX = []
for seq1 in testDataSeq:
	seqSimLi = []
	seqSimLi = FindCloset20Seq(seq1, testDataSeq)
	for amino in seq1:
		#做one hot encoding
		tmpLi = [0] * 20
		tmpLi[aminoDict[amino]] = 1
		testX = testX + tmpLi
	#存最後20個像的
	testX = testX + seqSimLi

with open('SPEuk.json', 'w') as outfile:  
	json.dump(testX, outfile)

testFile = r"C:\Users\MINHAN\Desktop\signalp\test\TMEuk.nr.fasta"
testDataSeq = ReadTestDataSeq(testFile)
#test data最像的20個
testX = []
for seq1 in testDataSeq:
	seqSimLi = []
	seqSimLi = FindCloset20Seq(seq1, testDataSeq)
	for amino in seq1:
		#做one hot encoding
		tmpLi = [0] * 20
		tmpLi[aminoDict[amino]] = 1
		testX = testX + tmpLi
	#存最後20個像的
	testX = testX + seqSimLi

with open('TMEuk.json', 'w') as outfile:  
	json.dump(testX, outfile)

localtime = time.asctime( time.localtime(time.time()) )
print("END time", localtime)

print("successful!")