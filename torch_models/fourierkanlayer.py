import torch as th
import numpy as np

# This implementation is taken from the GitHub user @GistNoesis
# Link to GitHub: https://github.com/GistNoesis/FourierKAN/

class NaiveFourierKANLayer(th.nn.Module):
    def __init__( self, inputdim, outdim, gridsize, addbias=True, smooth_initialization=False):
        super(NaiveFourierKANLayer,self).__init__()
        self.gridsize= gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim
        
        # With smooth_initialization, fourier coefficients are attenuated by the square of their frequency.
        # This makes KAN's scalar functions smooth at initialization.
        # Without smooth_initialization, high gridsizes will lead to high-frequency scalar functions,
        # with high derivatives and low correlation between similar inputs.
        grid_norm_factor = (th.arange(gridsize) + 1)**2 if smooth_initialization else np.sqrt(gridsize)
        
        #The normalization has been chosen so that if given inputs where each coordinate is of unit variance,
        #then each coordinates of the output is of unit variance 
        #independently of the various sizes
        self.fouriercoeffs = th.nn.Parameter( th.randn(2,outdim,inputdim,gridsize) / 
                                                (np.sqrt(inputdim) * grid_norm_factor ) )
        if( self.addbias ):
            self.bias  = th.nn.Parameter( th.zeros(1,outdim))

    #x.shape ( ... , indim ) 
    #out.shape ( ..., outdim)
    def forward(self,x):
        xshp = x.shape
        outshape = xshp[0:-1]+(self.outdim,)
        x = th.reshape(x,(-1,self.inputdim))
        #Starting at 1 because constant terms are in the bias
        k = th.reshape( th.arange(1,self.gridsize+1,device=x.device),(1,1,1,self.gridsize))
        xrshp = th.reshape(x,(x.shape[0],1,x.shape[1],1) ) 
        #This should be fused to avoid materializing memory
        c = th.cos( k*xrshp )
        s = th.sin( k*xrshp )
        #We compute the interpolation of the various functions defined by their fourier coefficient for each input coordinates and we sum them 
        y =  th.sum( c*self.fouriercoeffs[0:1],(-2,-1)) 
        y += th.sum( s*self.fouriercoeffs[1:2],(-2,-1))
        if( self.addbias):
            y += self.bias
        #End fuse
        '''
        #You can use einsum instead to reduce memory usage
        #It stills not as good as fully fused but it should help
        #einsum is usually slower though
        c = th.reshape(c,(1,x.shape[0],x.shape[1],self.gridsize))
        s = th.reshape(s,(1,x.shape[0],x.shape[1],self.gridsize))
        y2 = th.einsum( "dbik,djik->bj", th.concat([c,s],axis=0) ,self.fouriercoeffs )
        if( self.addbias):
            y2 += self.bias
        diff = th.sum((y2-y)**2)
        print("diff")
        print(diff) #should be ~0
        '''
        y = th.reshape( y, outshape)
        return y