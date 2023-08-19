import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
# Training data is 11 points in [0,1] inclusive regularly spaced
#train_x = torch.linspace(0, 1, 100)
# True function is sin(2*pi*x) with Gaussian noise
#train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2


# Load data from csv file
df = pd.read_csv('../dataset/socal/socal2.csv')
# normalize using MinMaxScaler
def normalize(df,columns):
    print('normalizing')
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    print(df[columns].head())
    print('normalizing done')
    return df

def standerdize(df,columns):
    print('standerdizing')
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    print(df[columns].head())
    print('standadizing done')
    return df


def preprocessing(dataset_path):
    df = pd.read_csv(dataset_path)
    print('preprocessing')
    print(df.columns)
    print(df.info())
    print(df.head())

    df = pd.read_csv(dataset_path)
    df = df.drop(columns=['street', 'citi','n_citi'],axis=0)
    df = df.dropna()
    df = df.reset_index(drop=True)
    df['price'] = df['price'].astype(np.double)
    standerdize(df,['sqft','price'])
    # normalize(df,['sqft','price'])
    df.to_csv('../dataset/df.csv', index=False)
    print(f'data shape: {df.shape}')
    print('preprocessing done')
    return df
    
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




class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(X_train, y_train, likelihood)

model = model.cuda()
likelihood = likelihood.cuda()


# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iter = 50
for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(X_train)
    # Calc loss and backprop gradients
    loss = -mll(output, y_train)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ))
    optimizer.step()
