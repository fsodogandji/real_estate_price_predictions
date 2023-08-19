import gpytorch
import pandas as pd
from sklearn.model_selection import train_test_split
from pytorch_lightning import Trainer
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models.exact_gp import   ExactGPModel 
from sklearn.model_selection import train_test_split
from gpytorch.mlls import NegativeMarginalLogLikelihood

# Load data from csv file
df = pd.read_csv('../dataset/socal/socal2.csv')

# Split data into training and test sets
train_df, test_df = train_test_split(df, test_size=0.2)

# Define the training data
X_train = train_df[['bath', 'sqft']].values
y_train = train_df['price'].values
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()

# Define the test data
X_test = test_df[['bath', 'sqft']].values
y_test = test_df['price'].values

X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()
if torch.cuda.is_available():
    X_train = X_train.to('cuda')
    y_train = y_train.to('cuda')
    X_test = X_test.to('cuda')
    y_test = y_test.to('cuda')

# Define the Gaussian process model
# Define the GP model
class GPModel(ExactGPModel):
    def __init__(self, train_x, train_y):
        super(GPModel, self).__init__(train_x, train_y)
        self.mean_module = ConstantMean()
        self.covar_module = RBFKernel()
        
    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return MultivariateNormal(mean, covar)
    
# Initialize the GP model
model = GPModel(X_train, y_train)
likelihood = GaussianLikelihood()

# Fit the GP model to the training data
model.train()
likelihood.train()
mll = ExactMarginalLogLikelihood(likelihood, model)
mll.train()

# Make predictions on the test data
model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    predictions = likelihood(model(X_test))
    mean = predictions.mean
    lower, upper = predictions.confidence_region()
    
# Print the confidence interval for the predictions
print("Confidence interval: [{:.2f}, {:.2f}]".format(lower.item(), upper.item()))

# Make predictions on test data
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x = torch.from_numpy(X_test.values).float()
    observed_pred = likelihood(model(test_x))
    mean = observed_pred.mean
    lower, upper = observed_pred.confidence_region()