import json
"""
with open('/Users/pipi9baby/Desktop/Data/trainX.json' , 'r') as reader:
	jf = json.loads(reader.read())

tmpLi = []
num = len(jf)
for i in range(0,num,700):
	tmpLi = tmpLi + jf[:700]
	jf = jf[720:]

with open('/Users/pipi9baby/Desktop/NoKnnData/trainX.json', 'w') as outfile:
	json.dump(tmpLi, outfile)


with open('/Users/pipi9baby/Desktop/Data/NCEuk.json' , 'r') as reader:
	jf = json.loads(reader.read())

tmpLi = []
num = len(jf)
for i in range(0,num,700):
	tmpLi = tmpLi + jf[:700]
	jf = jf[720:]

with open('/Users/pipi9baby/Desktop/NoKnnData/NCEuk.json', 'w') as outfile:
	json.dump(tmpLi, outfile)


with open('/Users/pipi9baby/Desktop/Data/SPEuk.json' , 'r') as reader:
	jf = json.loads(reader.read())

tmpLi = []
num = len(jf)
for i in range(0,num,700):
	tmpLi = tmpLi + jf[:700]
	jf = jf[720:]

with open('/Users/pipi9baby/Desktop/NoKnnData/SPEuk.json', 'w') as outfile:
	json.dump(tmpLi, outfile)
"""
with open('/Users/pipi9baby/Desktop/Data/TMEuk.json' , 'r') as reader:
	jf = json.loads(reader.read())

tmpLi = []
num = len(jf)
for i in range(0,num,700):
	tmpLi = tmpLi + jf[:700]
	jf = jf[720:]

with open('/Users/pipi9baby/Desktop/NoKnnData/TMEuk.json', 'w') as outfile:
	json.dump(tmpLi, outfile)

print("successfully!!")