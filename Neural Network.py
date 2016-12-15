import random
import math
import copy
def sigmoid(inX):
    try:
        denomiator=(1.0 + math.exp(-inX))
    except OverflowError:
        return 1.0-sigmoid(-inX)
    return 1.0 / denomiator

def Norm(vector):
    """
    vector[in] a vector represented as list/tuple or a number represented as int/float.
    return 2-norm of the vector
    """
    try:
        Norm=float(vector)
        if Norm<0:
            Norm=-Norm
    except TypeError:
        Norm=0
        for entry in vector:
            Norm+=entry**2
        Norm=math.sqrt(Norm)
    return Norm

def minmax(samples):
    min=samples[0][:]
    max=samples[0][:]
    for sample in samples:
        for i in range(len(sample)):
            if sample[i]<min[i]:
                min[i]=sample[i]
            if sample[i]>max[i]:
                max[i]=sample[i]
    return min,max

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
            return self.__input * self.__weight + self.__threshold
                


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


    def fit(self,input_data,output_data,num_hidden_layer=None,mean_size_hidden_layer=None,data_by_rows=True,test_data_ratio=0.2,max_step=200,node_error_epsilon=0.002,error=0.05):
        if num_hidden_layer == None:
            num_hidden_layer = int(math.sqrt(len(input_data[0]))) + 5
        if mean_size_hidden_layer == None:
            mean_size_hidden_layer = int(math.sqrt(len(input_data[0]) * len(output_data[0])))
        self.__generate(len(input_data[0]),len(output_data[0]),num_hidden_layer,mean_size_hidden_layer)
        #learning algorithm when there's no hidden layer
        #for i in range(len(self.__Neurons) - 1):#Layer by layer
        for i in range(1):#Change the Output layer Only
            if self.getError(input_data,output_data) < error:
                return
            for j in range(len(self.__Neurons[-1 - i])):#Neuron by neuron/Output by output
                NodeError = None
                DampingFactor = 0
                for step_count in range(max_step):#Step by step
                    if NodeError == None:
                        NodeError = self.getError(input_data,output_data,j)
                        print("Error of Node {}: {}".format(j,NodeError))
                    else:
                        NewNodeError = self.getError(input_data,output_data,j)
                        print("Error of Node {}: {}".format(j,NewNodeError))
                        print("Delta of Error:",NodeError - NewNodeError)
                        if NodeError >= NewNodeError:
                            if NodeError - NewNodeError < node_error_epsilon:
                                break
                            NodeError = NewNodeError
                        else:
                            print("Wrong Step, rolling back.")
                            self.__Neurons[-i - 1][j].adjustGeneralizedWeight(GeneralizedWeight)
                            DampingFactor+=1
                    grad = [0 for n_w in range(len(self.__Neurons[-2 - i]) + 1)]#gradient of Error with respect to generalizedweight.
                    for k in range(len(grad)):#Calculate gradient weight by weight(generalzide)
                        for l in range(len(output_data)):#sample by sample
                            self.setInput(input_data[l])
                            if k < (len(grad) - 1):#adjust kth weight
                                grad[k] = grad[k] + (self.__Neurons[-1 - i][j].getOutput() - output_data[l][j]) * 1 * self.__Neurons[-2 - i][k].getOutput()#falty?
                            else:#adjust threshold
                                grad[k] = grad[k] + (self.__Neurons[-1 - i][j].getOutput() - output_data[l][j]) * 1 * 1
                    GeneralizedWeight = self.__Neurons[-i - 1][j].getGeneralizedWeight()
                    NewGeneralizedWeight = [0] * len(grad)
                    for k in range(len(grad)):#Update weight by weight(Generalized)
                        NewGeneralizedWeight[k] = GeneralizedWeight[k] - self.__stepSize(DampingFactor) * grad[k]
                    #print("New weight:",Generalizedweight[:-1])
                    #print("New threshold:",Generalizedweight[-1])
                    self.__Neurons[-i - 1][j].adjustGeneralizedWeight(NewGeneralizedWeight)

    def test_fit(self,input_data,output_data,num_hidden_layer=None,mean_size_hidden_layer=None,data_by_rows=True,test_data_ratio=0.2,max_step=200,node_error_epsilon=0.002,error=0.05):
        if num_hidden_layer == None:
            num_hidden_layer = 1
        if mean_size_hidden_layer == None:
            mean_size_hidden_layer = int(math.sqrt(len(input_data[0]) + len(output_data[0]))) + 5
        self.__generate(len(input_data[0]),len(output_data[0]),num_hidden_layer,mean_size_hidden_layer)
        if len(input_data)!=len(output_data):
            raise NeuronException
        #adjust input layer:
        totalError=self.getError(input_data,output_data)
        for Epoch in range(40):
            print("Epoch:{}".format(Epoch))
            if totalError < error:
                print("Convergence reached. \nTotal rms error:{}".format(totalError))
                return
            #adjust weight ouf output layer
            print("Adjusting output layer... ")
            for output_i in range(len(self.__Neurons[-1])):
                Error = self.getError(input_data,output_data,output_i)
                print("Node {} Error:{}".format(output_i,Error))
                DampingFactor = 0
                rb_count=0#the count of consequtive rollbacks
                for step in range(max_step):
                    grad = [0] * (len(self.__Neurons[-2]) + 1)
                    for grad_i in range(len(grad)):#Calculate the gradient of E with respect to vector (w_1,w_2,...,w_n,theta)
                        for sample_i in range(len(input_data)):
                            self.setInput(input_data[sample_i])
                            if grad_i < len(grad)-1:
                                grad[grad_i]=grad[grad_i]+(self.__Neurons[-1][output_i].getOutput() - output_data[sample_i][output_i]) * 1 * self.__Neurons[-2][grad_i].getOutput()
                                #grad[k] = grad[k] + (self.__Neurons[-1 - i][j].getOutput() - output_data[l][j]) * 1 * self.__Neurons[-2 - i][k].getOutput()
                            else:
                                grad[grad_i]+=(self.__Neurons[-1][output_i].getOutput() - output_data[sample_i][output_i]) * 1 * 1
                    GeneralizedWeight = self.__Neurons[-1][output_i].getGeneralizedWeight()
                    NewGeneralizedWeight = [0] * len(GeneralizedWeight)
                    for gw_i in range(len(NewGeneralizedWeight)):
                        NewGeneralizedWeight[gw_i] = GeneralizedWeight[gw_i] - grad[gw_i] * self.__stepSize(DampingFactor)
                    self.__Neurons[-1][output_i].adjustGeneralizedWeight(NewGeneralizedWeight)
                    NewError = self.getError(input_data,output_data,output_i)
                    if NewError <= Error:
                        #print("New Error:{}:".format(NewError))
                        #print("Delta of Error:{}".format(NewError - Error))
                        if Error - NewError < node_error_epsilon:
                            break
                        Error = NewError
                        rb_count=0
                    else:
                        #print("Delta of Node Error{}:".format(NewError - Error))
                        #print("Over shooted.")
                        #print("Rolling back")
                        self.__Neurons[-1][output_i].adjustGeneralizedWeight(GeneralizedWeight)
                        DampingFactor+=1
                        rb_count+=1
                        if rb_count>=6:
                            break
            newTotalError=self.getError(input_data,output_data)
            print("Error:{}".format(newTotalError))
            
            #adjust weight ouf hidden layer
            if len(self.__Neurons)<3:
                continue
            print("Adjust Hidden layer... ")
            for hidden_i in range(len(self.__Neurons[-2])):
                Error = self.getError(input_data,output_data)
                print("Error:{}".format(Error))
                DampingFactor = 0
                rb_count=0#the count of consequtive rollbacks
                for step in range(max_step):
                    grad = [0] * (len(self.__Neurons[-3]) + 1)
                    for grad_i in range(len(grad)):#Calculate the gradient of E with respect to vector (w_1,w_2,...,w_n,theta)
                        for output_i in range(len(self.__Neurons[-1])):
                            for sample_i in range(len(input_data)):
                                self.setInput(input_data[sample_i])
                                if grad_i < len(grad)-1:
                                    grad[grad_i]=grad[grad_i]+(self.__Neurons[-1][output_i].getOutput() - 
                                        output_data[sample_i][output_i]) * self.__Neurons[-1][output_i].getWeight()[hidden_i] * \
                                        self.__Neurons[-2][hidden_i].getOutput()*(1-self.__Neurons[-2][hidden_i].getOutput()) * \
                                        self.__Neurons[-3][grad_i].getOutput()
                                        
                                else:
                                    grad[grad_i]=grad[grad_i]+(self.__Neurons[-1][output_i].getOutput() - 
                                        output_data[sample_i][output_i]) * self.__Neurons[-1][output_i].getWeight()[hidden_i] * \
                                        self.__Neurons[-2][hidden_i].getOutput()*(1-self.__Neurons[-2][hidden_i].getOutput()) * \
                                        1
                    GeneralizedWeight = self.__Neurons[-2][hidden_i].getGeneralizedWeight()
                    NewGeneralizedWeight = [0] * len(GeneralizedWeight)
                    for gw_i in range(len(NewGeneralizedWeight)):
                        NewGeneralizedWeight[gw_i] = GeneralizedWeight[gw_i] - grad[gw_i] * self.__stepSize(DampingFactor)
                    self.__Neurons[-2][hidden_i].adjustGeneralizedWeight(NewGeneralizedWeight)
                    NewError = self.getError(input_data,output_data)
                    #print("New Error{}:".format(NewError))
                    #print("Delta of Error:{}".format(NewError - Error))
                    if NewError <= Error:
                        if Error - NewError < node_error_epsilon:
                            break
                        Error = NewError
                        rb_count=0
                    else:
                        #print("Over shooted.")
                        #print("Rolling back")
                        self.__Neurons[-2][hidden_i].adjustGeneralizedWeight(GeneralizedWeight)
                        DampingFactor+=1
                        rb_count+=1
                        if rb_count>=6:
                            break
                newTotalError=self.getError(input_data,output_data)
                print("Error:{}".format(newTotalError))
            
            print("Delta of rms error during the epoch:{}".format(newTotalError-totalError))
            totalError=newTotalError
                        
    def test_fit2(self,input_data,output_data,num_hidden_layer=None,mean_size_hidden_layer=None,data_by_rows=True,test_data_ratio=0.2,max_step=200,error_epsilon=0.002,error=0.05):
        #batch gradient descent
        if num_hidden_layer == None:
            num_hidden_layer = 2
        if mean_size_hidden_layer == None:
            mean_size_hidden_layer = int(math.sqrt(len(input_data[0]) + len(output_data[0]))) + 5
        self.__generate(len(input_data[0]),len(output_data[0]),num_hidden_layer,mean_size_hidden_layer)
        print(self.generalizedWeightQuery())
        if len(input_data)!=len(output_data):
            raise NeuronException
        #adjust input layer:
        totalError=self.getError(input_data,output_data)
        DampingFactor=0
        for Epoch in range(max_step):
            if totalError<error:
                print("Convergence Reached. \nError:{}".format(totalError))
                return
            outputQuery=self.__outputQuery(input_data)
            errorQuery=self.__errorQuery(outputQuery,output_data)
            generalizedWeightQuery=self.__generalizedweightQuery()#Preserve the network in case of overflow or overstep
            for layer_i in range(len(self.__Neurons)-1,0,-1):
                for neuron_i in range(len(self.__Neurons[layer_i])):
                    GeneralizedWeight=self.__Neurons[layer_i][neuron_i].getGeneralizedWeight()
                    for gw_i in range(len(GeneralizedWeight)-1):
                        for sample_i in range(len(outputQuery)):
                            if layer_i==len(self.__Neurons)-1:#adjust output layer
                                GeneralizedWeight[gw_i]-=self.__stepSize(DampingFactor)*\
                                (outputQuery[sample_i][-1][neuron_i]-output_data[sample_i][neuron_i])*\
                                1*\
                                outputQuery[sample_i][layer_i-1][gw_i]#Adjust weight

                                GeneralizedWeight[-1]-=self.__stepSize(DampingFactor)*\
                                (outputQuery[sample_i][-1][neuron_i]-output_data[sample_i][neuron_i])*\
                                1*\
                                1#Adjust thresholds
                            else:#adjust hidden layer
                                GeneralizedWeight[gw_i]-=self.__stepSize(DampingFactor)*errorQuery[sample_i][layer_i-1][neuron_i]*\
                                outputQuery[sample_i][layer_i][neuron_i]*(1-outputQuery[sample_i][layer_i][neuron_i])*\
                                outputQuery[sample_i][layer_i-1][gw_i]#Adjust weights

                                GeneralizedWeight[-1]-=self.__stepSize(DampingFactor)*errorQuery[sample_i][layer_i-1][neuron_i]*\
                                outputQuery[sample_i][layer_i][neuron_i]*(1-outputQuery[sample_i][layer_i][neuron_i])*\
                                1#Adjust thresholds
                    self.__Neurons[layer_i][neuron_i].adjustGeneralizedWeight(GeneralizedWeight)
            newTotalError=self.getError(input_data,output_data)
            print("Epoch {} \nDelta:{} \nNew Error:{} ".format(Epoch,newTotalError-totalError,newTotalError))
            if newTotalError<=totalError:
                if totalError-newTotalError<error_epsilon:
                    print("Convergence reached. \nDelta:{} \nError:{} ".format(totalError-newTotalError,newTotalError))
                    return
                #DampingFactor=0
                totalError=newTotalError
            else:
                DampingFactor+=1
                print("DampingFactor:{}".format(DampingFactor))
                self.adjustAll(generalizedWeightQuery)#roll back
        print("Epoch limit reached. \nError:{} ".format(totalError))

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

    def __errorQuery(self,outputQuery,targetOutputDataset):
        """
        targetOutputDataset[in]:list of list
        errorQuery[out]: errorQuery[sampleIndex][layerIndex][error_index]
        returns a certain error of a certain neuron of a certain layer of a
        ceratain sample.
        ATTENTION!:layerIndex starts from 0:the first NON-INPUT LAYER(e.g hidden layer or output layer)
        """
        errorQuery=copy.deepcopy(outputQuery)
        for sampleIndex in range(len(errorQuery)):
            for layer_index in range(len(self.__Neurons)-1,0,-1):
                if layer_index==len(self.__Neurons)-1:
                    for output_index in range(len(self.__Neurons[-1])):
                        #errorQuery[sampleIndex][layer_index][output_index]=sigmoid(outputQuery[sampleIndex][-1][output_index])-\
                        #sigmoid(targetOutputDataset[sampleIndex][output_index])
                        #        #Apply sigmoid function and calculate the hallucinated output error. 
                        errorQuery[sampleIndex][layer_index][output_index]=(outputQuery[sampleIndex][-1][output_index]-
                        targetOutputDataset[sampleIndex][output_index])
                                
                else:
                    errorQuery[sampleIndex][layer_index][:]=self.__backPropagateError(layer_index+1,errorQuery[sampleIndex][layer_index+1])
            errorQuery[sampleIndex].pop(0)#Structure modification. No need to calculate the error of input layer.
        return errorQuery


    def __backPropagateError(self,layerIndex,Error):
        """
        Estimate the error of hidden layer
        Return the backpropogated error given the error of output of output layer
        parameters:

        """
        if len(Error) != len(self.__Neurons[layerIndex]):
            raise NeuronException
        if self.__Neurons[layerIndex - 1] is self.__Neurons[0]:
            raise NeuronException
        bpError = [0] * len(self.__Neurons[layerIndex - 1])
        for bp_i in range(len(bpError)):
            for error_i in range(len(Error)):
                bpError[bp_i]+=Error[error_i] * self.__Neurons[layerIndex][error_i].getWeight()[bp_i]
        return bpError

    def __generalizedweightQuery(self):
        generalizedweightQuery=[[]for i in range(len(self.__Neurons))]
        for layer_i in range(len(generalizedweightQuery)):
            for neuron_i in range(len(self.__Neurons[layer_i])):
                generalizedweightQuery[layer_i].append(self.__Neurons[layer_i][neuron_i].getGeneralizedWeight())
        return generalizedweightQuery


    def __stepSize(self,n):
        return 1 * math.exp(-0.5 * n)
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
        



    #def getMSSEError(self,InputData,OutputData,OutputIndex=None):
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
    #a = NeuralNetwork()
    #Input = [[random.random(),random.random(),random.random()] for i in range(30)]
    #Output = []
    #for input in Input:
    #    Output.append([input[0] + input[1] + input[2] + 10,input[1] + input[2] + 20,input[2] + 30])
    #a.test_fit2(Input,Output,num_hidden_layer=2,max_step=300,error_epsilon=1e-6,mean_size_hidden_layer=5)
    #print("Error:{}".format(a.getError(Input,Output)))
    #print("Test training data: ")
    #print("Test input:{}".format(Input[0]))
    #print("Network output:{}".format(a.getOutput(Input[0])))
    #print("Real output:{}".format([Input[0][0]+Input[0][1]+Input[0][2]+10,Input[0][1]+Input[0][2]+20,Input[0][2]+30]))
    #print("Test unseen data: ")
    #print("Test input:{}".format([1,1,1]))
    #print("Network output:{}".format(a.getOutput([1,1,1])))
    #print("Real output:{}".format([13,22,31]))
    #del a
#    b = NeuralNetwork()
#    Input = [[random.random(),random.random()] for i in range(200)]
#    Output = []
#    for input in Input:
#        Output.append([5*math.sin(input[0]+input[1])])
#    b.test_fit2(Input,Output,num_hidden_layer=1,max_step=300,error_epsilon=1e-8)
#    print("Error:{}".format(b.getError(Input,Output)))
#    print("Test training data: ")
#    print("Test input:{}".format(Input[0]))
#    print("Network output:{}".format(b.getOutput(Input[0])))
#    print("Real output:{}".format(math.sin(Input[0][0]+Input[0][1])))
#    print("Test unseen data: ")
#    print("Test input:{}".format([1,1]))
#    print("Network output:{}".format(b.getOutput([1,1])))
#    print("Real output:{}".format(math.sin(1+1)))
#    del b
    c = NeuralNetwork()
    Input = [[i*0.1-5] for i in range(100)]
    Output = []
    for input in Input:
        Output.append([10*math.exp(-(input[0])**2)])
    c.test_fit2(Input,Output,max_step=500,num_hidden_layer=2,mean_size_hidden_layer=10,error_epsilon=1e-8,error=1e-2)
    print(c.generalizedWeightQuery())
    print("Error:{}".format(c.getError(Input,Output)))
    import numpy as np
    import seaborn as sns
    from matplotlib import pyplot as plt
    #Input = [[i*0.01-5] for i in range(1000)]
    Output = []
    netOutput=[]
    for input in Input:
        Output.append([10*math.exp(-(input[0])**2)])
        netOutput.append(c.getOutput(input)[0])
    npInput=np.array(Input);
    npRealOutput=np.array(Output)
    npNetOutput=np.array(netOutput)
    plt.figure(figsize=(8,5))
    plt.plot(npInput,npRealOutput, label='npRealOutput')
    plt.plot(npInput,npNetOutput, label='npNetOutput')
    plt.legend()
    plt.show()
    del c
if __name__ == "__main__":
    test()



        

