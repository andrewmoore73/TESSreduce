import numpy as np


def _sigma_mask(data,sigma=3):
	"""
	Just does a sigma clip on an array.
	
	Parameters
	----------
	data : array
		A single image 
	sigma : float
		sigma used in the sigma clipping

	Returns
	-------
	clipped : array
		A boolean array to mask the original array
	"""
	
	clipped = ~sigma_clip(data,sigma=sigma).mask
	return clipped 


def _strip_units(data):
	"""
 	Converts non array_like objects to their bare values in a numpy ndarray

  	Parameters
   	----------
    	data : TYPE
  		object to strip (unless its array_like in which case nothing happens)

     	Returns
      	-------
       	data : numpy ndarray
		The values of the data parameter converted into a numpy ndarray
  	"""
  
	if type(data) != np.ndarray:
		data = data.value
	return data



def grads_rad(flux):
	"""
	Computes the radius of a point in the flux gradient vs flux gradient gradient 2d space

  	Parameters
   	----------
    	flux : array_like
     		flux array

	Returns
 	-------
  	rad : array_like
   		radius of a point in the flux gradient vs flux gradient gradient 2d space (same dimensions as the flux parameter)
  	"""
	
	rad = np.sqrt(np.gradient(flux)**2+np.gradient(np.gradient(flux))**2)
	return rad

def grad_flux_rad(flux):
	"""
 	Computes the radius of a point in the flux vs flux gradient 2d space

   	Parameters
    	----------
     	flux : array_like
      		flux array

 	Returns
  	-------
   	rad : array_like
    		radius of a point in the flux vs flux gradient 2d space (same dimensions as the flux parameter)
        """
	
	rad = np.sqrt(flux**2+np.gradient(flux)**2)
    	return rad
