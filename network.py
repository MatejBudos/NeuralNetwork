import numpy as np
import random
from utils import *
import time
import datetime
from activations import *

class Layer:
    def __init__( self, input_size, output_size, activation : Activation  ) -> None:
        self.input_size = input_size
        self.output_size = output_size

        #Xavier weights initialization 
        limit = np.sqrt(1 / (input_size + output_size))
        self.weights = np.random.uniform(-limit, limit, size = ( output_size, input_size ) )

        self.bias = np.random.randn( output_size, 1 )

        #activation function of the layer
        self.activation  = activation

        self.output = None
        self.input = None
        
    def compute_output( self, input ):
        #computes output of the layer with added bias and activation function
        self.input = input
        self.output = self.activation.f( np.dot( self.weights, input ) + self.bias )
        return self.output
    
    def backwards( self, output_gradient, learning_rate ):
        #back propagation with gradient descent
        
        #calculate output error between network's output and actual output
        output_gradient = output_gradient * self.activation.derivative( self.output )
        weights_gradient = np.dot( output_gradient, self.input.T )

        #output gradient for layer before this one
        input_gradient = np.dot( self.weights.T, output_gradient )

        #update weights
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient
    


    def export_weights_and_biases( self ):
        #exports weights,biases and activation function in form:
        # w w w
        # w w w
        # w w w
        #
        # b
        # b
        # b
        #Activation function

        result = ""
        for neuron in self.weights:
            for weight in neuron:
                result += str( weight ) + " "
            result += "\n"
        result += "\n"
        for biases in self.bias:
            for bias in biases:
                result += str( bias ) + " "
            result += "\n"
        result += self.activation.__str__()
        return result
    


    def import_weights_and_biases( self, weights, biases ):
        #sets parsed weights and biases from load weights
        self.weights = weights
        self.bias = biases 
    def __str__( self ):
        #string representation of the layer in form:
        # input_size -> output_size : activation_function
        return f'{ self.input_size } -> { self.output_size } : { self.activation }\n'



    
class Model:
    def __init__( self, network = None ) -> None:
        # list of layers 
        self.network = network

    def train( self, epochs, learning_rate, train_input, train_output, validation_input, validation_output ):
        #num of train data
        count = len( train_input )

        #for Restoring best model weights
        min_error = np.inf
        min_network = None
        min_epoch = 0

        start_time = time.time()
        epoch = 0

        #training by time (5min)
        #while time.time() - start_time < 300:
        #training by epochs     
        while epoch < epochs:

            for i in np.random.permutation( count ):
                #takes random [[x],[y]] coordinate and predicts its output
                output = self.predict( train_input[ i ] )

                #output error 
                grad = self.mse_prime( train_output[ i ], output )
                #backpropagation
                for layer in reversed( self.network ):
                    grad = layer.backwards( grad, learning_rate )
            #MSE
            e = self.evaluate( validation_input, validation_output)


            #Saving best weights
            if e < min_error:
                min_error = e
                min_network = self.network
                min_epoch = epoch
            
            if ( epoch + 1 ) % 10 == 0: 
                print( "Epoch: ", epoch + 1, end = " " )
                print( "Mse: ",round( e , 5 ) )
            
            epoch += 1
        
        print( f"Stopped after { epoch } epochs" )
        print( f"Restored model from epoch { min_epoch }" )
        self.network = min_network

    def predict( self, x ):
        #takes the input throught whole network and return the output 
        output = x
        for layer in self.network:
            output = layer.compute_output( output )
        return output
    
    def evaluate( self, input, output ):
        error = 0
        for x, y in zip( input, output ):
            output = self.predict( x )
            error += self.mse( y, output )
        return error / len( input )
    
    def mse( self, y, d ):
        return np.mean( ( d - y ) ** 2 )

    def mse_prime( self, y_true, y_pred):
        return 2 * (y_pred - y_true) / np.size(y_true)
    
    def save_weights( self, error  ):
        #exports all layers in model
        #and saves them into one txt file
        #file_name format: hour-minute-MSE-error value
        t = datetime.datetime.now()
        h = t.hour
        m = t.minute
        file_name = f"{ h }-{ m }-MSE-{ round( error, 4 ) }.txt"
        with open( file_name, "w" ) as fw:
            for layer in self.network:
                export = layer.export_weights_and_biases()
                fw.write( export )
                fw.write( "\n")
        return file_name

    def load_weights( self, file_name ):
        #wights + bias + activation function parser from text file
        weights = []
        biases = []
        self.network = []
        activations = { "Sigmoid": Sigmoid(), "Tanh":Tanh(), "Linear":Linear(), "ReLu":ReLu() }
        loading_biases = False
        with open( file_name, 'r' ) as fr:
            for line in fr:
                line = line.strip()
                if line in activations.keys():
                    loading_biases = False
                    weights = np.array( weights )
                    biases = np.array( biases )
                    input_size = weights.shape[ 1 ]
                    output_size = weights.shape[ 0 ]
                    tmp = Layer( input_size, output_size, activations[ line ] )
                    tmp.import_weights_and_biases( weights, biases )
                    self.network.append( tmp )
                    weights = []
                    biases = []
                    continue
                if line == "":
                    loading_biases = True
                    continue
                line = line.split()
                element = np.array( list( map( lambda x : float( x ), line ) ) )

                if loading_biases:
                    biases.append( element  )
                else:
                    weights.append( element )
                
    def __str__( self ) -> str:
        #string representation of the model
        result = ""
        for layer in self.network:
            result += layer.__str__()
        return result

def split_data( data ):
    #shuffles and splits data into train and validation sets
    np.random.shuffle( data )
    index = int( len( data ) * 0.8 )
    train = data[ : index ]
    validation = data[ index: ]
    return train, validation

def input_output_split( data ):
    #splits data set to input and output sets
    input = np.array( [ element[ :2 ] for element in data ] )
    output = np.array( [ element[ 2: ] for element in data ] )
    return input, output
def reshape( data ):
    #reshpaes data from [x, y] to [[x],[y]]
    return np.reshape( data, ( len( data ), 2, 1 ) )

#Test function
def test( model, file_name ):
    data = np.loadtxt( file_name )
    inp, out = input_output_split( data )
    inp = reshape( inp )
    error = model.evaluate( inp, out )
    print( f"MSE: {error}")


if __name__ == "__main__":
    file_name = "mlp_train.txt"
    data = np.loadtxt( file_name )
    train, validation = split_data( data )
    train_input, train_output = input_output_split( train )
    validation_input, validation_output = input_output_split( validation )
    train_input = reshape( train_input )
    validation_input = reshape( validation_input )
    l1 = Layer( 2, 32, Sigmoid() )
    l2 = Layer( 32, 32, Sigmoid() )
    l3 = Layer( 32, 32, Sigmoid() ) 
    l4 = Layer( 32, 1, Linear() ) 

    network = [ l1, l2, l3, l4 ]
    epochs = 1000
    learning_rate = 0.01
    


    
    model = Model( network )

    #Training
    model.train( epochs, learning_rate, train_input, train_output, validation_input, validation_output )
    error = model.evaluate( validation_input, validation_output )
    print( f"Mse: { error }" )
    name = model.save_weights( error )
    print(f"Model saved as: {name}")
    '''
    #Test
    model_file = "22-21-MSE-0.0283.txt"
    model.load_weights( model_file )
    test_file = "mlp_train.txt"
    test( model, test_file )
    print( model )
    '''