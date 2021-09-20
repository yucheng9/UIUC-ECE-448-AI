import numpy as np

class NaiveBayes(object):
	def __init__(self,num_class,feature_dim,num_value):
		"""Initialize a naive bayes model. 

		This function will initialize prior and likelihood, where 
		prior is P(class) with a dimension of (# of class,)
			that estimates the empirical frequencies of different classes in the training set.
		likelihood is P(F_i = f | class) with a dimension of 
			(# of features/pixels per image, # of possible values per pixel, # of class),
			that computes the probability of every pixel location i being value f for every class label.  

		Args:
		    num_class(int): number of classes to classify
		    feature_dim(int): feature dimension for each example 
		    num_value(int): number of possible values for each pixel 
		"""

		self.num_value = num_value
		self.num_class = num_class
		self.feature_dim = feature_dim

		self.prior = np.zeros((num_class))
		self.likelihood = np.zeros((feature_dim,num_value,num_class))

	def train(self,train_set,train_label):
		""" Train naive bayes model (self.prior and self.likelihood) with training dataset. 
			self.prior(numpy.ndarray): training set class prior (in log) with a dimension of (# of class,),
			self.likelihood(numpy.ndarray): traing set likelihood (in log) with a dimension of 
				(# of features/pixels per image, # of possible values per pixel, # of class).
			You should apply Laplace smoothing to compute the likelihood. 

		Args:
		    train_set(numpy.ndarray): training examples with a dimension of (# of examples, feature_dim)
		    train_label(numpy.ndarray): training labels with a dimension of (# of examples, )
		"""

		# YOUR CODE HERE

		# Calculate likelihood 
		for label in range(len(train_label)):
			# Calculate priors
			self.prior[train_label[label]] += 1
			for pixel in range(len(train_set[label])):
				# print("Training")
				# Calculate likelihood 
				self.likelihood[pixel, int(train_set[label][pixel]), train_label[label]] += 1

		# Laplace smoothing
		# for k in range (0.1, 10, 0.1):
		k = 1
		for i in range(self.num_class):
			# print("Smoothing")
			self.likelihood[:,:,i] = (self.likelihood[:,:,i]+k)/(self.prior[i]+self.num_value*k)

		# Take the log to avoid underflow during testing
		self.likelihood = np.log(self.likelihood)
		self.prior = np.log(self.prior/len(train_label)) 
		print("End of training")
		pass

	def test(self,test_set,test_label):
		""" Test the trained naive bayes model (self.prior and self.likelihood) on testing dataset,
			by performing maximum a posteriori (MAP) classification.  
			The accuracy is computed as the average of correctness 
			by comparing between predicted label and true label. 

		Args:
		    test_set(numpy.ndarray): testing examples with a dimension of (# of examples, feature_dim)
		    test_label(numpy.ndarray): testing labels with a dimension of (# of examples, )

		Returns:
			accuracy(float): average accuracy value  
			pred_label(numpy.ndarray): predicted labels with a dimension of (# of examples, )
		"""    

		# YOUR CODE HERE

		accuracy = 0
		pred_label = np.zeros((len(test_set)))

		'''
		posterior_prob_matrix = np.ndarray((len(test_set), self.num_class))

		for i in range(len(test_set)): 
			for j in range(self.num_class): 
				print("Testing")
				posterior_prob_matrix[i][j] = self.prior[j]
				posterior_prob_matrix[i][j] += np.sum(self.likelihood[np.arange(self.feature_dim),int(test_set[i][j]),j])
			pred_label[i] = np.argmax(posterior_prob_matrix[i])

		accuracy = (len(test_set) - np.count_nonzero(pred_label - test_label))/len(test_set)
		'''
		
		for i in range(len(test_label)): 
 			evaluation = np.zeros(self.num_class) 
 			for j in range(self.num_class): 
 				evaluation[j] = self.prior[j] 
 				for k in range(self.feature_dim): 
 					evaluation[j] += self.likelihood[k, int(test_set[i][k]), j] 
 			pred_label[i] = np.argmax(evaluation) 
 			if (pred_label[i] == test_label[i]): accuracy += 1 
		
		accuracy /= len(test_set)  
		print("Accuracy:", accuracy)
		print("End of testing")
		return accuracy, pred_label

	def save_model(self, prior, likelihood):
		""" Save the trained model parameters 
		"""    

		np.save(prior, self.prior)
		np.save(likelihood, self.likelihood)

	def load_model(self, prior, likelihood):
		""" Load the trained model parameters 
		""" 

		self.prior = np.load(prior)
		self.likelihood = np.load(likelihood)

	def intensity_feature_likelihoods(self, likelihood):
	    """
	    Get the feature likelihoods for high intensity pixels for each of the classes,
	        by sum the probabilities of the top 128 intensities at each pixel location,
	        sum k<-128:255 P(F_i = k | c).
	        This helps generate visualization of trained likelihood images. 
	    
	    Args:
	        likelihood(numpy.ndarray): likelihood (in log) with a dimension of
	            (# of features/pixels per image, # of possible values per pixel, # of class)
	    Returns:
	        feature_likelihoods(numpy.ndarray): feature likelihoods for each class with a dimension of
	            (# of features/pixels per image, # of class)
	    """
	    # YOUR CODE HERE
	    print("Get the feature likelihoods for high intensity pixels")
	    return np.sum(np.exp(likelihood[:,128:,:]), axis = 1)