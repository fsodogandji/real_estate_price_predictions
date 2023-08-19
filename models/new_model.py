import pandas as pd
from torch.utils.data import DataLoader, Dataset, Subset, random_split
import os
from PIL import Image
import numpy as np
import torch
from torch import tensor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RepeatedKFold, cross_val_score
import xgboost
from numpy import absolute
from torchmetrics.functional import mean_absolute_percentage_error



class RealEstateDataset(Dataset):
    def __init__(self, csv_path, to_tensor=True):
        self.data = pd.read_csv(csv_path)
        self.X = self.data.drop(columns=['price', 'zpid']).values
        self.y = self.data['price'].values
        if to_tensor:
            self.X = tensor(self.X, dtype=torch.float32)
            self.y = tensor(self.y, dtype=torch.float32)
        else:
            self.X = self.X.astype(np.float32)
            self.y = self.y.astype(np.float32)    
        #self.X = tensor(self.X, dtype=torch.float32)
        #self.y = tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        #self.X = tensor(self.X, dtype=torch.float32)
        return  self.X[idx], self.y[idx]

    def group_kfold(self, n_splits):
        gkf = GroupKFold(n_splits=n_splits)
        groups = self.raw['Patient']
        for train_idx, val_idx in gkf.split(self.raw, self.raw, groups):
            train = Subset(self, train_idx)
            val = Subset(self, val_idx)
            yield train, val

    def group_split(self, test_size=0.2):
        """To test no-kfold
        """
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size)
        groups = self.raw['Patient']
        idx = list(gss.split(self.raw, self.raw, groups))
        train = Subset(self, idx[0][0])
        val = Subset(self, idx[0][1])
        return train, val    



real_estate_dataset = RealEstateDataset(csv_path='../dataset/df.csv')
real_estate_dataloader = DataLoader(real_estate_dataset, batch_size=32, shuffle=True)



class RealEstateImageDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        return image



real_estate_image_dataset = RealEstateImageDataset(image_dir='../dataset/processed_images')
real_estate_image_dataloader = DataLoader(real_estate_image_dataset, batch_size=32, shuffle=True)





class RealEstateCombinedDataset(Dataset):
    def __init__(self, csv_path, image_dir):
        self.data = pd.read_csv(csv_path)
        self.X = self.data.drop(columns=['price', 'zpid'])
        self.y = self.data['price']
        self.image_dir = image_dir
        self.image_paths = {f: os.path.join(image_dir, f) for f in os.listdir(image_dir)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_id = self.data.iloc[idx]['id']
        image = Image.open(self.image_paths[image_id])
        return self.X.iloc[idx], image, self.y.iloc[idx]

real_estate_combined_dataset = RealEstateCombinedDataset(csv_path='../dataset/df.csv',
                                                         image_dir='../dataset/processed_images')
real_estate_combined_dataloader = DataLoader(real_estate_combined_dataset, batch_size=32, shuffle=True)


import pytorch_lightning as pl
import torch
import torch.nn as nn

class RealEstateRegressionModel(pl.LightningModule):
    def __init__(self, real_estate_dataloader):
        super().__init__()
        self.dataloader = real_estate_dataloader
        self.linear = nn.Linear(in_features=5, out_features=1)  # assuming the input size is 5 (latitude, longitude, beds, baths, area)

    def forward(self, X):

        return self.linear(X)

    def training_step(self, batch, batch_idx):
        X, y = batch
        y = y.unsqueeze(1)
        y_pred = self.forward(X)
        loss = nn.MSELoss()(y_pred, y)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def train_dataloader(self):
        return self.dataloader

def conv_block(input_size, output_size):
    block = nn.Sequential(
        nn.Conv2d(input_size, output_size, (3, 3)), nn.ReLU(), nn.BatchNorm2d(output_size), nn.MaxPool2d((2, 2)),
    )

    return block

class LitClassifier(pl.LightningModule):
    def __init__(
        self, lr: float = 1e-3, num_workers: int = 4, batch_size: int = 32,
    ):
        super().__init__()
        self.lr = lr
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.conv1 = conv_block(3, 16)
        self.conv2 = conv_block(16, 32)
        self.conv3 = conv_block(32, 64)

        self.ln1 = nn.Linear(64 * 26 * 26, 16)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(16)
        self.dropout = nn.Dropout2d(0.5)
        self.ln2 = nn.Linear(16, 5)

        self.ln4 = nn.Linear(5, 10)
        self.ln5 = nn.Linear(10, 10)
        self.ln6 = nn.Linear(10, 5)
        self.ln7 = nn.Linear(5, 1)

    def forward(self, tab):
        tab = self.ln4(tab)
        tab = self.relu(tab)
        tab = self.ln5(tab)
        tab = self.relu(tab)
        tab = self.ln6(tab)
        tab = self.relu(tab)

        x = self.relu(tab)

        return self.ln7(x)

    def training_step(self, batch, batch_idx):
        tabular, y = batch

        criterion = torch.nn.L1Loss()
        y_pred = torch.flatten(self(tabular))
        y_pred = y_pred.double()

        loss = criterion(y_pred, y)
        mape = mean_absolute_percentage_error(y_pred, y)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "MAPE": mape, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        tabular, y = batch

        criterion = torch.nn.L1Loss()
        y_pred = torch.flatten(self(tabular))
        y_pred = y_pred.double()

        val_loss = criterion(y_pred, y)
        mape = mean_absolute_percentage_error(y_pred, y)
        return {"val_loss": val_loss, "MAPE": mape}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_mape = torch.stack([x["MAPE"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"val_loss": avg_loss, "MAPE":avg_mape,"log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        tabular, y = batch

        criterion = torch.nn.L1Loss()
        y_pred = torch.flatten(self(tabular))
        y_pred = y_pred.double()

        test_loss = criterion(y_pred, y)

        return {"test_loss": test_loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        logs = {"test_loss": avg_loss}
        return {"test_loss": avg_loss, "log": logs, "progress_bar": logs}

    def setup(self, stage):

        image_data = RealEstateDataset(csv_path='../dataset/df.csv')

        train_size = int(0.80 * len(image_data))
        val_size = int((len(image_data) - train_size) / 2)
        test_size = int((len(image_data) - train_size) / 2)

        self.train_set, self.val_set, self.test_set = random_split(image_data, (train_size, val_size, test_size))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=(self.lr))

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)        

real_estate_model = RealEstateRegressionModel(real_estate_dataloader)
real_estate = next(iter(real_estate_dataloader))

print("toto",real_estate,"tata")

real_estate_model = LitClassifier()
trainer = pl.Trainer(accelerator='gpu',max_epochs=100)
#trainer.fit(real_estate_model)
data = RealEstateDataset(csv_path='../dataset/df.csv')


#################
#folds = data.group_kfold(num_kfolds)
#
#for fold, (trainset, valset) in enumerate(folds):
#    dataloaders = {
#        'train': DataLoader(trainset, batch_size=batch_size,
#                            shuffle=True, num_workers=2),
#        'val': DataLoader(valset, batch_size=batch_size,
#                          shuffle=False, num_workers=2)
#    }
#####################    

model_xgboost = xgboost.XGBRegressor(colsample_bytree=0.4,
                 gamma=0,                 
                 learning_rate=0.07,
                 max_depth=3,
                 min_child_weight=1.5,
                 n_estimators=100,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42) 
real_estate_dataset = RealEstateDataset(csv_path='../dataset/df.csv',to_tensor=False)
real_estate_dataloader = DataLoader(real_estate_dataset, batch_size=32, shuffle=True)

real_estate_data = next(iter(real_estate_dataloader))
print(real_estate_data[1].numpy())
X = real_estate_data[0].numpy()
y = real_estate_data[1].numpy()

# Define the callback function
#callback = xgboost.callback.

model_xgboost.fit(X, y, verbose=True)
model_xgboost.save_model("xgboost.model")
accuray = model_xgboost.score(X,y)
print(accuray)


cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model_xgboost, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
# force scores to be positive
scores = absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()) )