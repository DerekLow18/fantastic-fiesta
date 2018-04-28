from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
#import scipy.spatial
import copy
import string

#making sure it works on a smaller example
'''
dataset =[[0,0],[1,0]]
hidden_layer_weights = [[0.15,0.25],[0.20,0.30]]
output_layer_weights = [[0.40,0.50],[0.45,0.55]]
'''
hidden_layer_bias = 0.35
output_layer_bias = 0.6

A = ord('A')
def convertCharToFloats(inputArray):
	targetFloats = np.zeros(len(inputArray))
	for c in range(len(inputArray)):
		if 'A' <= inputArray[c] <= 'Z':
			#tempSum += ord(c) - A + 1
			#print(ord(inputArray[c]) - A+1)
			targetFloats[c] = round(float((ord(inputArray[c]) - A+1))/26,2)-.01
			#print(targetFloats[c])
	return targetFloats

def convertFloatsToChar(inputArray):
	newString = []
	for c in range(len(inputArray)):
		#print(inputArray[c])
		val = math.ceil(inputArray[c]*26)
		#print(val)
		char = str(chr(int(inputArray[c]*26)+A))
		#print(char)
		newString.append(char)
	return newString

targetString = ["H","E","L","L","O","W","O","R","L","D"]

targetFloats = convertCharToFloats(targetString)
initial = np.round(np.random.rand(len(targetFloats)),2)
print(targetFloats)
print(convertFloatsToChar(targetFloats))
dataset = [initial,targetFloats]
#print(dataset)
hidden_layer_weights = np.random.rand(len(targetFloats),len(targetFloats))
output_layer_weights = np.random.rand(len(targetFloats),len(targetFloats))
learningRate = 0.5

#error calculation between the predicted step and the actual step, euclidean distance
def error(prediction, actual):
	return scipy.spatial.distance.euclidean(prediction, actual)

def squaredError(prediction,actual):
	squaredErrorVector = []
	for index in range(len(prediction)):
		squaredErrorVector.append((1/2)*(actual[index] - prediction[index])**2)
	return np.sum(squaredErrorVector)

#formula for the prediction of what the next step will look like.
#Currently, it's at sigmoid function
def activation(activity):
	return round(1 / (1 + math.exp(-activity)),9)

def pdSquaredError(predicted, actual):
	return round(-(actual - predicted),9)

def pdEuclideanDistance(predicted,actual):
#calculate value for partial deriv of euclidean distance w.r.t. predicted
	return (predicted-actual)/(np.sqrt((predicted-actual)**2))

#partial derivative of the activation function
def pdSigmoid(x):
	return round(x*(1-x),9)

#using this for the purposes of testing a standard prediction formula
def prediction(timeStep):
	#first calculate the hidden layer values, called prediction array
	predictionArray = np.zeros(len(output_layer_weights))
	predictionActivity = np.zeros(len(output_layer_weights))
	for hiddenIndex in range(len(predictionArray)):
		predicted = 0
		for inputIndex in range(len(dataset[0])):
			predicted += dataset[0][inputIndex]*hidden_layer_weights[inputIndex][hiddenIndex]
		predicted = predicted + hidden_layer_bias
		predictionActivity[hiddenIndex] = round(predicted,9)
		predictionArray[hiddenIndex] = round(activation(predicted),9)
	'''
	now we calculate the values for the output array
	'''
	outputArray = np.zeros(len(output_layer_weights))
	outputActivity = np.zeros(len(output_layer_weights))
	for outputIndex in range(len(outputArray)):
		predicted = 0
		for hiddenIndex in range(len(predictionArray)):
			predicted += predictionArray[hiddenIndex]*output_layer_weights[hiddenIndex][outputIndex]
		predicted = predicted + output_layer_bias
		outputActivity[outputIndex] = round(predicted,9)
		outputArray[outputIndex] = round(activation(predicted),9)

	#print(predictionArray)
	#print(outputArray)
	return [predictionArray, outputArray, predictionActivity, outputActivity]


'''
change the weight between one source neuron and the target neuron

'''
def weightChangeOutput(predicted,actual,priorStep):
	#print("priorStep is",priorStep)
	i = round(pdSquaredError(predicted,actual),9)
	#print("the pd squared error is ", i)
	#partial derivative of euclidean distance with respect to the prediction
	#print("the activity is", predicted)
	j = round(pdSigmoid(predicted),9)
	#print("the pd sigmoid is",j)
		#partial derivative of activation function with respect to the activity
	totalChange = round(i*j*priorStep,9)
	return totalChange
'''
calculate the weight change for one weight in the hidden layer

Prior step is the input to the hidden layer
'''
def weightChangeHidden(predictedArray,actualArray,priorStep,weightIndex,hiddenOutput):
	totalError = 0
	#here, predictedArray is the array from the layer forward to the current hidden layer
	#actual array is the corresponding output array
	#print("THe predicted array is ",predictedArray)
	#print("the actual array is ", actualArray)
	for outputIndex in range(len(predictedArray)):
		i = round(pdSquaredError(predictedArray[outputIndex],actualArray[outputIndex]),9)
		j = round(pdSigmoid(predictedArray[outputIndex]),9)
		w = output_layer_weights[weightIndex][outputIndex]
		#print(i,j,w)
		totalError = totalError + round(i*j*w,9)
	hO = round(pdSigmoid(hiddenOutput),9)
	#print(totalError,hO,priorStep)
	totalChange = round(totalError*hO*priorStep,9)
	#print(totalChange)
	return totalChange



#main network training function
def trainNetwork(Max_iters = 1000):
	#predictionMatrix = []
	#Iterates through all values in the data set
	#for i in range(len(dataset)):
		#predict the value for the next step and store it
	i=0
	while (i <Max_iters):
		predictionMatrix = prediction(dataset[0]);#store the predictions for the array into a matrix
		'''
		now that we have the predictions, we need to calculate the weight change for each weight in the
		weight matrix. Start with the output layer's weights from the hidden layer
		'''
		global output_layer_weights,hidden_layer_weights
		#print("Before: \n", output_layer_weights,"\n",hidden_layer_weights,"\n")
		updatedWeights = copy.deepcopy(output_layer_weights)
		for weightArrayIndex in range(len(output_layer_weights)):
			for weightValueIndex in range(len(output_layer_weights[weightArrayIndex])):
				weightValue = output_layer_weights[weightArrayIndex][weightValueIndex]
				weightDelta=weightChangeOutput(predictionMatrix[1][weightValueIndex],dataset[1][weightValueIndex],predictionMatrix[0][weightArrayIndex])
				updatedWeights[weightArrayIndex][weightValueIndex] = round(weightValue - learningRate*weightDelta,9)
		#print(updatedWeights)
		'''
		now we do a similar thing for the input to hidden layer weights.
		'''
		updatedIToHWeights = copy.deepcopy(output_layer_weights)
		for weightArrayIndex in range(len(hidden_layer_weights)):
			for weightValueIndex in range(len(hidden_layer_weights[weightArrayIndex])):
				weightValue = hidden_layer_weights[weightValueIndex][weightArrayIndex]
				weightDelta=weightChangeHidden(predictionMatrix[1],dataset[1],dataset[0][weightValueIndex],weightArrayIndex,predictionMatrix[0][weightArrayIndex])
				updatedIToHWeights[weightValueIndex][weightArrayIndex] = round(weightValue - learningRate * weightDelta,9)
		#print(updatedIToHWeights)
		output_layer_weights = updatedWeights
		hidden_layer_weights = updatedIToHWeights
		print(squaredError(predictionMatrix[1],dataset[1]))
		currGuess = convertFloatsToChar(predictionMatrix[1])
		print(''.join(currGuess))
		if currGuess == targetString:
			exit()

		i += 1
	#print("After: \n",output_layer_weights,"\n",hidden_layer_weights,"\n")
	print(predictionMatrix[1])
	print(''.join(convertFloatsToChar(predictionMatrix[1])))


trainNetwork()
