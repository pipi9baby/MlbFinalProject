#動態規劃法，輸入兩個string回傳一個int
def dynamicString(self,str1, str2):
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
