import random
import math
def sigmoid(inX):  
    return 1.0 / (1 + math.exp(-inX)) 
class Neuron():
    def __init__(self,Input=None,weight=None,threshold=0):
        if Input != None:
            if type(Input) == list:#When input is a list of Neurons adjust settings for hidden layers or output
                                   #layers
                self.__input = Input[:]
                self.__threshold = float(threshold)
                if weight == None:
                    self.__weight = [random.random() for i in range(len(Input))]
                elif len(weight) != len(Input):
                    self.__weight = [0 for i in range(len(Input))]
                    raise NeuronException
                else:
                    self.__weight = weight[:]
            else:#Adjust settings for input layer
                self.__input = float(Input)
                self.__weight = float(weight)
                self.__threshold = 0.0
        else:#settings for input layers
            self.__input = 0.0
            self.__weight = 1.0
            self.__threshold = 0.0# No threshold for input layer by default
        
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
        
    def getOutput(self):
        if type(self.__input) == list:#Get output for hidden layers or output layer
            Output = 0
            for i in range(len(self.__input)):
                Output+=self.__weight[i] * self.__input[i].getOutput()
            Output+=self.__threshold
            #Output = sigmoid(Output)
            return Output
        else:#Get output for input layer
            return self.__input * self.__weight + self.__threshold
    def __del__(self):
        print("Bye")

class NeuralNetwork():
    def __init__(self):
        self.__Neurons = []
    def __generate(self,Num_Input,Num_Output,Num_Hidden):
            self.__Neurons.append([Neuron() for i in range(Num_Input)])
            for i in range(Num_Hidden):
                self.__Neurons.append([Neuron(Input=[self.__Neurons[-1][:]],threshold=0) for i in range(10)])
            self.__Neurons.append([Neuron(Input=self.__Neurons[-1][:],threshold=0) for i in range(Num_Output)])

    def fit(self,input_data,output_data,num_hidden_layer=None,data_by_rows=True,TestDataRatio=0.2,max_step=200,Epsilon=0.02):
        if num_hidden_layer == None:
            num_hidden_layer = int(math.sqrt(len(input_data[0]))) + 5

        num_hidden_layer = 0# algorithm for hidden layers to be implemented


        self.__generate(len(input_data[0]),len(output_data[0]),num_hidden_layer)
        #learning algorithm when there's no hidden layer
        for i in range(len(self.__Neurons) - 1):#Layer by layer
            for j in range(len(self.__Neurons[-1 - i])):#Neuron by neuron/Output by output
                for step_count in range(max_step):#Step by step
                    grad = [0 for n_w in range(len(self.__Neurons[-2 - i]) + 1)]#gradient of Error with respect to generalizedweight.
                    for k in range(len(grad)):#Calculate gradient weight by weight(generalzide)
                        for l in range(len(output_data)):#sample by sample
                            self.setInput(input_data[l])
                            if k < (len(grad) - 1):#adjust kth weight
                                grad[k] = grad[k] + (self.__Neurons[-1 - i][j].getOutput() - output_data[l][j]) * 1 * self.__Neurons[-2 - i][k].getOutput()#falty?
                            else:#adjust threshold
                                grad[k] = grad[k] + (self.__Neurons[-1 - i][j].getOutput() - output_data[l][j]) * 1 * 1
                                #grad[k]=0
                    Generalizedweight = self.__Neurons[-i - 1][j].getGeneralizedWeight()
                    for k in range(len(grad)):#Updae weight by weight(Generalized)
                        Generalizedweight[k] = Generalizedweight[k] - self.__stepSize(step_count) * grad[k]
                    #print("New weight:",Generalizedweight[:-1])
                    #print("New threshold",Generalizedweight[-1])
                    self.__Neurons[-i - 1][j].adjustGeneralizedWeight(Generalizedweight)

    def __stepSize(self,n):
        #return 0.02*math.exp(-5*n)
        return 0.005

    def setInput(self,data):
        if len(data) == len(self.__Neurons[0]):
            for i in range(len(data)):
                self.__Neurons[0][i].setInput(data[i])
            return
        raise NeuronException

    def getOutput(self,Input=None):
        if Input != None:
            self.setInput(Input)
        return [Node.getOutput() for Node in self.__Neurons[-1]]

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
    Input = [[random.random(),random.random(),random.random()] for i in range(200)]
    Output = []
    for input in Input:
        Output.append([input[0] + input[1] + input[2] + 10,input[1] + input[2] + 10,input[2] + 10])
    #y=2x1+3x2
    a.fit(Input,Output,num_hidden_layer=0,max_step=300)
    print(a.getOutput([1,1,1]))
    del a
if __name__ == "__main__":
    test()



        

