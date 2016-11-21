import random
import math
import copy
def sigmoid(inX):  
    return 1.0 / (1 + math.exp(-inX))


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
                    self.__weight = [random.random() for i in range(len(Input))]
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
        if len(GeneralizedWeight) != (len(self.__weight) + 1):
            raise NeuronException
        self.adjustWeight(GeneralizedWeight[:-1])
        self.adjustThreshold(GeneralizedWeight[-1])

    def getWeight(self):
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
        print("An",self.__type,"Neuron says goodbye!")

class NeuralNetwork():
    def __init__(self):
        self.__Neurons = []


    def __generate(self,Num_Input,Num_Output,Num_Hidden_Layer,mean_size_hidden_layer):
            self.__Neurons.append([Neuron("Input") for i in range(Num_Input)])
            for i in range(Num_Hidden_Layer):
                if mean_size_hidden_layer<=0:
                    raise NeuronException
                self.__Neurons.append([Neuron(Type="Hidden",Input=self.__Neurons[-1][:],Threshold=random.random()) for j in range(mean_size_hidden_layer)])
            self.__Neurons.append([Neuron(Type="Output",Input=self.__Neurons[-1][:],Threshold=0) for i in range(Num_Output)])


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
        for Epoch in range(100):
            if self.getError(input_data,output_data) < error:
                return
            #adjust weight ouf output layer
            print("Adjusting output layer")
            for output_i in range(len(self.__Neurons[-1])):
                Error = self.getError(input_data,output_data,output_i)
                print("Node {} Error{}:".format(output_i,Error))
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
                        print("Delta of Node Error{}:".format(NewError - Error))
                        print("Over shooted.")
                        print("Rolling back")
                        self.__Neurons[-1][output_i].adjustGeneralizedWeight(GeneralizedWeight)
                        DampingFactor+=1
                        rb_count+=1
                        if rb_count>=6:
                            break
            print("Error{}".format(self.getError(input_data,output_data)))
                        
            #adjust weight ouf hidden layer
            if len(self.__Neurons)<3:
                continue
            print("Adjust Hidden layer.")
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
                print("Error:{}".format(self.getError(input_data,output_data)))

                        

                
            
           

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
        InputBacup=self.getInput()#backup input
        outputQuery=[[]for i in range(len(inputDataSet))]
        for sampleIndex in range(len(inputDataSet)):
            outputQuery[sampleIndex].append(self.getOutput(inputDataSet[sampleIndex],0))#log output of input layer
            for layerIndex in range(1,len(self.__Neurons)):#log ouput of other layers
                outputQuery[sampleIndex].append([self.__Neurons[layerIndex][outputIndex].getOutput(Input=outputQuery[sampleIndex][-1]) for outputIndex in range(len(self.__Neurons[layerIndex]))])
                #outputQuery[sampleIndex].append(self.getOutput(Input=inputDataSet[sampleIndex],LayerIndex=layerIndex))
                if __name__ =="__main__":
                    Output1=outputQuery[sampleIndex][layerIndex]
                    Output2=self.getOutput(Input=inputDataSet[sampleIndex],LayerIndex=layerIndex)
                    if Output1!=Output2:
                        print("Falty")
        self.setInput(InputBacup)#restore input
        return outputQuery

    def __errorQuery(self,outputQuery,targetOutputDataset):
        """
        targetOutputDataset[in]:list of list
        outputQuery[out]:list of list of list
        """
        errorQuery=[[] for i in len(outputQuery)]
        for sampleIndex in range(len(inputDataSet)):
            pass


    def __backPropagateError(self,layerIndex,OutputError):
        """
        Estimate the error of hidden layer
        Return the backpropogated error given the error of output of output layer
        parameters:
        layerIndex[in]: the index of layer(must be a hidden layer)
        OutputError[in]: the error vector of output layer
        """
        Error = OutputError[:]
        for layer_i in range(len(self.__Neurons)-1,layerIndex,-1):
            if self.__Neurons[layer_i] is self.__Neurons[-1]:
                for i in range(len(Error)):
                    Error[i] = sigmoid(Error[i])
            bpError=[0]*len(self.__Neurons[layer_i-1])
            for i in range(len(bpError)):
                for j in range(len(self.__Neurons[layer_i])):
                    bpError[i]+=self.__Neurons[layer_i][j].getWeight()[i]*Error[j]
            Error=bpError[:]
        return bpError




    def __stepSize(self,n):
        return 0.05 * math.exp(-2 * n)
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
        Return the mean(with respect to the number of samples) of the sum of squares of errors. 
        If OutputIndex==None, return the Error of the whole output, otherwise return the error of a specific output.
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
            Error = Error / len(InputData)
        else:
            for i in range(len(InputData)):
                Error+=math.pow(self.getOutput(InputData[i])[OutputIndex] - OutputData[i][OutputIndex],2)
            Error = Error / len(InputData)
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
    a = NeuralNetwork()
    Input = [[random.random(),random.random(),random.random()] for i in range(30)]
    Output = []
    for input in Input:
        Output.append([input[0] + input[1] + input[2] + 10,input[1] + input[2] + 20,input[2] + 30])
    a.test_fit(Input,Output,num_hidden_layer=1,max_step=100,node_error_epsilon=1e-4,mean_size_hidden_layer=2)
    print("Error: {}".format(a.getError(Input,Output)))
    print(a.getOutput([1,1,1]))
    print(a.getOutput(LayerIndex=-1))
    del a
    #b = NeuralNetwork()
    #Input = [[random.random(),random.random()] for i in range(200)]
    #Output = []
    #for input in Input:
    #    Output.append([5*math.sin(input[0]+input[1])+5])
    #b.test_fit(Input,Output,num_hidden_layer=1,mean_size_hidden_layer=3,max_step=30,node_error_epsilon=1e-4)
    #print("Error: {}".format(b.getError(Input,Output)))
    #print(b.getOutput([1,1]))
    #print(math.sin(1+1))
    #del b
if __name__ == "__main__":
    test()



        

