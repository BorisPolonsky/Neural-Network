import random
import math
import copy

def sigmoid(inX):
    try:
        denominator=(1.0 + math.exp(-inX))
    except OverflowError:
        return 1.0-sigmoid(-inX)
    return 1.0 / denominator


class Neuron():
    def __init__(self,Type="Input",Input=None,Weight=None,Threshold=0):
        """
        type[in] "Input" "Hidden" or else, case else will be considered as "Output"
        """
        if Type == "Input":#Initiate a neuron for input
            if Input != None:
                try:
                    float(Input)
                except TypeError:
                    raise NeuronException
                self.__input = float(Input)
            else:
                self.__input = 0.0
            self.__weight = 1.0#No amplification for input
            self.__threshold = 0.0# No threshold for input layer
            self.__type = Type
            return
        elif Type == "Hidden":
            self.__type = Type
        elif Type == "Output":
            self.__type = Type
        else:
            raise NeuronException()
        if Input != None:#Initiate a neuron in hidden/output layer layer
            if type(Input) == list:#Input must be a list of Neuron objects for neurons in hidden/Output layer
                self.__input = Input[:]
                self.__threshold = float(Threshold)
                if Weight == None:
                    self.__weight = [2*random.random()-1 for i in range(len(Input))]
                elif len(Weight) != len(Input):
                    self.__weight = [0 for i in range(len(Input))]
                    raise NeuronException
                else:
                    self.__weight = Weight[:]
            else:
                raise NeuronException
        else:
            raise NeuronException


    def adjustWeight(self,weight):
        if type(self.__input) == list:#Adjust weight for hidden layers or output layer
            if len(self.__input) == len(weight):
                self.__weight[:] = weight
            else:
                raise NeuronException
        else:#Adjust weight for input layer
            self.__weight = float(weight)


    def adjustThreshold(self,threshold):
        self.__threshold = float(threshold)


    def adjustGeneralizedWeight(self,GeneralizedWeight):#Generalized weight=weight+[threshold]
        if self.__type=="Input":
            if len(GeneralizedWeight)!=2:
                raise NeuronException
            self.adjustWeight(GeneralizedWeight[0]) 
            self.adjustThreshold(GeneralizedWeight[1])
        else:
            if len(GeneralizedWeight) != (len(self.__weight) + 1):
                raise NeuronException
            self.adjustWeight(GeneralizedWeight[:-1])
            self.adjustThreshold(GeneralizedWeight[-1])


    def getWeight(self):
        if self.__type=="Input":
            return self.__weight
        else:
            return self.__weight[:]

    def getThreshold(self):
        return self.__threshold


    def getGeneralizedWeight(self):
        Generalizedweight = self.getWeight()
        if type(Generalizedweight) == list:
            Generalizedweight+=[self.getThreshold()]
        else:
            Generalizedweight = [Generalizedweight] + [self.getThreshold()]
        return Generalizedweight


    def setInput(self,Input):
        if type(self.__weight) == list and type(Input) == list and len(self.__weight) == len(Input):
            self.__input[:] = Input
            return
        else:
            self.__input = float(Input)
            return
        raise NeuronException
        

    def getInput(self):
        if type(self.__input)==list:
            return self.__input[:]
        else:
            return self.__input


    def getOutput(self,Input=None):
        """
        If Input is given, then return the output with respect to given input without
        modifying the NeuralNetwork, including the original inputs of it. 
        Input[in]: list of float.
        """
        if type(self.__input) == list:#Get output for hidden layers or output layer
            Output = 0
            if Input == None:
                for i in range(len(self.__input)):
                    Output+=self.__weight[i] * self.__input[i].getOutput()
            else:
                if(len(self.__weight) == len(Input)):
                    for i in range(len(Input)):
                        Output+=self.__weight[i] * Input[i]
                else:
                    raise NeuronException
            Output+=self.__threshold
            if(self.__type == "Hidden"):
                Output = sigmoid(Output)
            return Output
        else:#Get output for input layer
            if Input == None:
                Input = self.__input
            return float(Input) * self.__weight + self.__threshold
                

    def __del__(self):
        print("A" if self.__type=="Hidden" else "An",self.__type,"Neuron says goodbye!")


class NeuralNetwork():
    def __init__(self):
        self.__Neurons = []


    def __generate(self,Num_Input,Num_Output,Num_Hidden_Layer,mean_size_hidden_layer):
            self.__Neurons.append([Neuron("Input") for i in range(Num_Input)])
            for i in range(Num_Hidden_Layer):
                if mean_size_hidden_layer<=0:
                    raise NeuronException
                weight_init=[1.0/math.sqrt(len(self.__Neurons[-1]))*(random.random()*2-1) for i in range(len(self.__Neurons[-1]))]
                #W~U[-1/sqrt(n),1/sqrt(n)], U is a uniform distribution. n is the number neuron in the last layer(e.g number of input in this neuron). 
                #<Understanding the difficulty of training deep feedforward neural networks> by Xavier Glorot & Yoshua Bengio
                self.__Neurons.append([Neuron(Type="Hidden",Input=self.__Neurons[-1][:],Weight=weight_init,Threshold=0) for j in range(mean_size_hidden_layer)])
            self.__Neurons.append([Neuron(Type="Output",Input=self.__Neurons[-1][:],Threshold=0) for i in range(Num_Output)])
            

    def __rescaleInput(self,samples):
        """
        Adjust the weights and thresholds in input layers to rescales the input of the network. 
        samples[in]:list of input vectors
        """
        min,max=self.__minmax(samples)
        #The input layer rescales each imput. Each output of input layer will be ranged from -5 to 5. 
        for input_i in range(len(self.__Neurons[0])):
            scale=max[input_i]-min[input_i]
            mean=(max[input_i]+min[input_i])/2.0
            if scale==0:
                self.__Neurons[0][input_i].adjustGeneralizedWeight([1,-mean])
            else:
                self.__Neurons[0][input_i].adjustGeneralizedWeight([10.0/scale,-mean*10.0/scale])


    def __minmax(self,samples):
        min=samples[0][:]
        max=samples[0][:]
        for sample in samples:
            for i in range(len(sample)):
                if sample[i]<min[i]:
                    min[i]=sample[i]
                if sample[i]>max[i]:
                    max[i]=sample[i]
        return min,max


    def adjustAll(self,generalizedWeightQuery):
        for layer_i in range(len(generalizedWeightQuery)):
            for neuron_i in range(len(generalizedWeightQuery[layer_i])):
                self.__Neurons[layer_i][neuron_i].adjustGeneralizedWeight(generalizedWeightQuery[layer_i][neuron_i])


    def generalizedWeightQuery(self):
        WeightQuery=[]
        for layer in self.__Neurons:
            WeightQuery.append([])
            for neuron in layer:
                WeightQuery[-1].append(neuron.getGeneralizedWeight())
        return WeightQuery


    def fit(self,input_data,output_data,num_hidden_layer=None,mean_size_hidden_layer=None,data_by_rows=True,test_data_ratio=0.2,max_step=20000,error_epsilon=0.002,error=0.05):
        #batch gradient descent
        if num_hidden_layer == None:
            num_hidden_layer = 2
        if mean_size_hidden_layer == None:
            mean_size_hidden_layer = int(math.sqrt(len(input_data[0]) + len(output_data[0]))) + 5
        self.__generate(len(input_data[0]),len(output_data[0]),num_hidden_layer,mean_size_hidden_layer)
        #print(self.generalizedWeightQuery())
        if len(input_data)!=len(output_data):
            raise NeuronException
        #adjust input layer:
        totalError=self.getError(input_data,output_data)
        minErr=totalError
        DampingFactor=0
        errorLog=[minErr]
        finalNetworkParam=self.generalizedWeightQuery()
        for Epoch in range(max_step):
            if totalError<error:
                print("Convergence Reached. \nError:{}".format(totalError))
                return errorLog
            outputQuery=self.__outputQuery(input_data)
            deltaQuery=self.__deltaQuery(outputQuery,output_data)
            generalizedWeightQuery=self.__generalizedweightQuery()#Preserve the network in case of overflow or overstep
            for layer_i in range(len(self.__Neurons)-1,0,-1):
                for neuron_i in range(len(self.__Neurons[layer_i])):
                    GeneralizedWeight=self.__Neurons[layer_i][neuron_i].getGeneralizedWeight()
                    for sample_i in range(len(outputQuery)):
                        for gw_i in range(len(GeneralizedWeight)-1):
                            GeneralizedWeight[gw_i]-=self.__stepSize(DampingFactor)*deltaQuery[sample_i][layer_i-1][neuron_i]*\
                            outputQuery[sample_i][layer_i-1][gw_i]#Adjust weight
                        GeneralizedWeight[-1]-=self.__stepSize(DampingFactor)*\
                        deltaQuery[sample_i][layer_i-1][neuron_i]*\
                        1*\
                        1#Adjust threshold
                    self.__Neurons[layer_i][neuron_i].adjustGeneralizedWeight(GeneralizedWeight)
            newTotalError=self.getError(input_data,output_data)
            #print("Epoch {} \nDelta:{} \nNew Error:{} ".format(Epoch,newTotalError-totalError,newTotalError))
            errorLog.append(newTotalError)
            if newTotalError<=totalError:
                if newTotalError<minErr:
                    minErr=newTotalError
                    finalNetworkParam=self.__generalizedweightQuery()
                if totalError-newTotalError<error_epsilon:
                    print("Convergence reached. \nDelta:{} \nError:{} ".format(totalError-newTotalError,minErr))
                    self.adjustAll(finalNetworkParam)
                    return errorLog
                #DampingFactor=0
                totalError=newTotalError
            else:
                if newTotalError-totalError>1:#Allow error to ascend by margin 1
                    DampingFactor+=1
                    #print("DampingFactor:{}".format(DampingFactor))
                    self.adjustAll(generalizedWeightQuery)#roll back
                else:
                    totalError=newTotalError
        print("Epoch limit reached. \nError:{} ".format(minErr))
        self.adjustAll(finalNetworkParam)
        return errorLog


    def __outputQuery(self,inputDataSet):
        """
        return each output of each layer of each input data. 
        Parameters
        inputDataset[in]:list of list
        outputQuery[out]:list of list of list

        inputDataset[sample_index][input_index] is a certain input of a certain sample. 
        inputDataset[sample_index] is a certain input vector(list) of a certain sample.
        outputQuery[sample_index][layer_index][output_index] returns a certain output given
        sample_index,layer_index,output_index
        """
        InputBackup=self.getInput()#backup input
        outputQuery=[[]for i in range(len(inputDataSet))]
        for sampleIndex in range(len(inputDataSet)):
            outputQuery[sampleIndex].append(self.getOutput(inputDataSet[sampleIndex],0))#log output of input layer
            for layerIndex in range(1,len(self.__Neurons)):#log ouput of other layers
                outputQuery[sampleIndex].append([self.__Neurons[layerIndex][outputIndex].getOutput(Input=outputQuery[sampleIndex][-1]) for outputIndex in range(len(self.__Neurons[layerIndex]))])#lower computation cost
                #outputQuery[sampleIndex].append(self.getOutput(Input=inputDataSet[sampleIndex],LayerIndex=layerIndex))
                if __name__ =="__main__":
                    Output1=outputQuery[sampleIndex][layerIndex]
                    Output2=self.getOutput(Input=inputDataSet[sampleIndex],LayerIndex=layerIndex)
                    if Output1!=Output2:
                        print("Falty")
                        raise NeuronException
        self.setInput(InputBackup)#restore input
        return outputQuery


    def __deltaQuery(self,outputQuery,targetOutputDataset):
        """
        targetOutputDataset[in]:list of list
        deltaQuery[out]: deltaQuery[sampleIndex][layerIndex][error_index]
        returns a certain delta of a certain neuron of a certain layer of a
        ceratain sample.
        ATTENTION!:layerIndex starts from 0:the first NON-INPUT LAYER(e.g hidden layer or output layer)
        """
        deltaQuery=copy.deepcopy(outputQuery)
        #calculate delta for output layers
        for sample_index in range(len(outputQuery)):
            for output_index in range(len(outputQuery[sample_index][-1])):
                deltaQuery[sample_index][-1][output_index]=outputQuery[sample_index][-1][output_index]-targetOutputDataset[sample_index][output_index]#delta equals to error for output layers
        #backpropogate delta
        for sample_index in range(len(deltaQuery)):
            for layer_index in range(len(self.__Neurons)-2):
                for output_index in range(len(self.__Neurons[-2-layer_index])):
                    delta=0
                    for nl_output_index in range(len(self.__Neurons[-1-layer_index])):
                        Weight=self.__Neurons[-1-layer_index][nl_output_index].getWeight()
                        delta+=deltaQuery[sample_index][-1-layer_index][nl_output_index]*Weight[output_index]
                    delta=outputQuery[sample_index][-2-layer_index][output_index]*(1-outputQuery[sample_index][-2-layer_index][output_index])*delta
                    deltaQuery[sample_index][-2-layer_index][output_index]=delta
            deltaQuery[sample_index].pop(0)#Structure modification. No need to calculate the delta of input layer.
        return deltaQuery


    def __generalizedweightQuery(self):
        generalizedweightQuery=[[]for i in range(len(self.__Neurons))]
        for layer_i in range(len(generalizedweightQuery)):
            for neuron_i in range(len(self.__Neurons[layer_i])):
                generalizedweightQuery[layer_i].append(self.__Neurons[layer_i][neuron_i].getGeneralizedWeight())
        return generalizedweightQuery


    def __stepSize(self,n):
        return 10 * math.exp(-0.5 * n)
        #return 0.005


    def setInput(self,data):
        if len(data) == len(self.__Neurons[0]):
            for i in range(len(data)):
                self.__Neurons[0][i].setInput(data[i])
            return
        raise NeuronException


    def getInput(self):
        return [neuron.getInput() for neuron in self.__Neurons[0]]


    def getOutput(self,Input=None,LayerIndex=None):
        """
        Input[in]: the input of NeuralNetwork (e.g. input of input layer)
        LayerIndex[in]: specify the layer(LayerIndex==0 suggests the input layer)
        """
        if LayerIndex == None:
            LayerIndex = len(self.__Neurons) - 1
        if Input != None:
            self.setInput(Input)
        #for i in range(LayerIndex + 1):
        #    if i == 0:
        #        ret = [Node.getOutput() for Node in self.__Neurons[i]]
        #    else:
        #        ret = [Node.getOutput(Input=ret) for Node in self.__Neurons[i]]
        #if __name__ == "__main__":
        #    new_ret=[neuron.getOutput() for neuron in self.__Neurons[LayerIndex]]
        #    if new_ret!=ret:
        #        print("Falty")
        #return ret
        ret=[]
        for neuron in self.__Neurons[LayerIndex]:
            ret.append(neuron.getOutput())
        return ret
        

    def getError(self,InputData,OutputData,OutputIndex=None):
        """
        Return the rms error. 
        If OutputIndex==None, return the rms error of the whole output, otherwise return the rms error of a specific output.
        Output Index starts from 0.
        """
        if len(InputData) != len(OutputData):
            raise NeuronException
        Error = 0
        if OutputIndex == None:
            for i in range(len(InputData)):          
                Output = self.getOutput(InputData[i])
                for j in range(len(Output)):
                    Error+=math.pow(Output[j] - OutputData[i][j],2)
            Error = Error / (len(InputData) * len(self.__Neurons[-1]))
            Error = math.sqrt(Error)
        else:
            for i in range(len(InputData)):
                Error+=math.pow(self.getOutput(InputData[i])[OutputIndex] - OutputData[i][OutputIndex],2)
            Error = Error / len(InputData)
            Error = math.sqrt(Error)
        return Error


    def size(self):
        size = 0
        for layer in self.__Neurons:
            size+=len(layer)
        return size
        

    def __del__(self):
        for layer in self.__Neurons:
            for item in layer:
                del item


class NeuronException(Exception):
    def __init__(self):
        pass


def test():
    import numpy as np
    import seaborn as sns
    from matplotlib import pyplot as plt

    net = NeuralNetwork()
    Input=[]
    Output=[]
    for x_2 in range(11):
        for x_1 in range(11):
            Input.append([x_1*0.1-0.5,x_2*0.1-0.5])
    for input in Input:
        Output.append([(input[0])**2+(input[1])**2])
        #Output.append([10])
    errLog=net.fit(Input,Output,max_step=100000,num_hidden_layer=1,mean_size_hidden_layer=5,error_epsilon=1e-20,error=1e-4)
    print(net.generalizedWeightQuery())
    print("Error:{}".format(net.getError(Input,Output)))

    y=np.array(errLog)
    x=np.array([i for i in range(len(y))])
    plt.figure(figsize=(8,5))
    plt.plot(x,y, label='Emprical Risk')
    plt.legend()
    plt.show()

    fig = plt.figure()
    ax = Axes3D(fig)
    netOutput=[]
    for input in Input:
        netOutput.append(net.getOutput(input)[0])
    npNetOutput=np.array(netOutput)
    ax.plot_surface(X, Y, npNetOutput.reshape(11,11), rstride=1, cstride=1, cmap='rainbow')
    ax.scatter(np.array(Input).T[0],np.array(Input).T[1],npNetOutput,)
    ax.set_zlabel('NetOutput') 
    ax.set_ylabel('x_1')
    ax.set_xlabel('x_2')
    plt.show()
if __name__ == "__main__":
    test()



        

