import numpy as np
class Activation:
    def f( x ):
        return 0
    def derivative( output ):
        return 0
    def __str__( self ) -> str:
        return ""
    
class Sigmoid( Activation ):

    def f( self, x ):
        return 1 / ( 1 + np.e ** ( -x ) )
    def derivative( self, x ):
        return x * ( 1 - x )
    def __str__( self ) -> str:
        return "Sigmoid"

class Tanh( Activation ):
    def f( self, x ):
        return np.tanh( x )
    def derivative( self, x ):
        return 1 - np.tanh( x ) ** 2
    def __str__( self ) -> str:
        return "Tanh"

class ReLu( Activation ):
    def f( self, x ):
        return np.maximum( 0, x )
    def derivative( self, x ):
        return np.where( x > 0, 1, 0 )
    def __str__( self ) -> str:
        return "ReLu"

class Linear( Activation ):
    def f( self, x ):
        return x
    def derivative( self, x ):
        return 1
    def __str__( self ) -> str:
        return "Linear"