from collections import defaultdict

#動態規劃法，輸入兩個string回傳一個int
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

testFile = r"C:\Users\MINHAN\Desktop\NCEuk.nr.fasta"
trainFile = r"C:\Users\MINHAN\Desktop\train.fasta"

aminoNum = 35

#讀取train data的資料
fr = open(trainFile)
train = fr.read()
fr.close()
trainLi = train.split(">")
trainDataSeq = []
trainY = []
for eachData in trainLi:
	if eachData == "":
		continue
	
	tmpLi = eachData.split("\n")
	trainDataSeq.append(tmpLi[1][:aminoNum])
	if "S" in tmpLi[2]:
		trainY.append(1)
	else:
		trainY.append(0)

print("training資料已讀取")
#讀取testDataSeq的資料
fr = open(testFile)
test = fr.read()
fr.close()
testLi = test.split(">")
testDataSeq = []
for eachData in testLi:
	if eachData == "":
		continue
	
	tmpLi = eachData.split("\n")
	testDataSeq.append(tmpLi[1][:aminoNum])

print("test資料已讀取")
#取train data最像的20個
seqSimLi = []
for seq1 in trainDataSeq:
	simDict = defaultdict(list)
	for seq2 in trainDataSeq:
		if seq1 == seq2:
			continue
		
		returnInt = dynamicString(seq1, seq2)
		simDict[returnInt].append(seq2)
	
	#找最大的20個
	items = list(simDict.keys())
	items.sort()
	limit = 0
	for key in items:
		value = simDict[key]
		for eachSeq in value:
			index = trainDataSeq.index(eachSeq)
			seqSimLi.append(trainY[index])
			limit += 1
			if(limit > 20):
				break
		if(limit > 20):
			break

print("KNN前20資料已讀取")
#做trainData 的 one hot encoding
aminoDict ={'G':0,'A':1,'V':2,'L':3,'I':4,'F':5,'W':6,'Y':7,'D':8,'H':9,'N':10,'E':11,'K':12,'Q':13,'M':14,'R':15,'S':16,'T':17,'C':18,'P':19}
trainX = []
for seq in trainDataSeq:
	for amino in seq:
		#做one hot encoding
		tmpLi = [0] * 20
		tmpLi[aminoDict[amino]] = 1
		trainX = trainX + tmpLi
	#存最後20個像的
	trainX = trainX + seqSimLi[:20]
	seqSimLi = seqSimLi[20:]

#做trainData 的 one hot encoding
testX = []
for seq in trainDataSeq:
	for amino in seq:
		#做one hot encoding
		tmpLi = [0] * 20
		tmpLi[aminoDict[amino]] = 1
		testX = testX + tmpLi

print(testX)
print(trainX)
print(trainY)
print("successful!")