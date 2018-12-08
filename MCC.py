# -*- coding: UTF-8 -*-
import math

TP = 0; FN = 0; TN = 0; FP = 0;
for i in range(len(predictY)):
    if(predictY[i] == 1 and y_Test[i] == 1): TP += 1
    elif(predictY[i] == 1 and y_Test[i] == 0): FN += 1
    elif(predictY[i] == 0 and y_Test[i] == 0): TN += 1
    elif(predictY[i] == 0 and y_Test[i] == 1): FP += 1

MCC = ((TP * TN) - (FP * FN)) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
print("Accurany : %s" %MCC)
