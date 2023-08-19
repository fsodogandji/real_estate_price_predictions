import glob
import os

import albumentations as A
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from aim.pytorch_lightning import AimLogger
from albumentations.pytorch import ToTensorV2
from PIL import Image
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger
from skimage import io, transform
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split
from torchinfo import summary
from torchmetrics.functional import mean_absolute_percentage_error
from torchvision import transforms, utils


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
    # standerdize(df,['sqft','price'])
    # normalize(df,['sqft','price'])
    # df.to_csv('../dataset/df.csv', index=False)
    print(f'data shape: {df.shape}')
    print('preprocessing done')
    return df

def check_dataset(dataset):
    for i in range(10):
        image,house_features,y = dataset[i]
        print(i, image.shape, house_features.shape, y.shape)


def get_train_test(dataset, split_percentage):
    train_percentage, test_percentage = (split_percentage)
    train_size = int(train_percentage * len(dataset))
    test_size = int(test_percentage * len(dataset))
    val_size = len(dataset) - train_size - test_size

    train_set, test_set, val_set = random_split(dataset, [train_size, test_size, val_size])
    return train_set, test_set, val_set            


def get_summary(model,dataset,batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # get the first batch of data
    first_batch = next(iter(dataloader))[1]
    print('first batch',first_batch.shape)
    summary(model, input_size=(batch_size, 1, 28,28,3))
    #summary(model, *first_batch.shape)

class RealEstateCombinedDataset(Dataset):
    def __init__(self, df, image_dir, transform = None, logging = False):

        self.data = df
        self.y = self.data['price']
        self.image_dir = image_dir
        self.image_paths = {f: os.path.join(image_dir, f) for f in os.listdir(image_dir)}
        self.transform = transform
        self.logging = logging

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):  
        image_id = self.data.iloc[idx]['image_id']
       # print(self.image_paths)
        #print('Image ID: ',int((image_id)))
        image = Image.open(self.image_paths[f'{int(image_id)}.jpg'])
        image = image.convert('RGB')
        if self.logging:
            print(self.image_paths[f'{int(image_id)}.jpg'])
            print("Image format:", image.format)
            print("Mode:", image.mode)
            print("Data type:", image.info)
            print("Size:", image.size)
        image = np.array(image).astype(np.float32)
        house_features_column = set(self.data.columns) - set(['image_id', 'price'])
        house_features = self.data[list(house_features_column)].iloc[idx]
        y = self.y.iloc[idx]
        #sample = {'image': image, 'house_features': house_features, 'y': y}
        # sample = {'image': image, 'house_features': house_features, 'y': y}
        # sample['image'] = transforms.ToTensor()(sample['image'])
        # sample['house_features'] = torch.from_numpy(sample['house_features'].values)
        # sample['y'] = torch.from_numpy(np.array([sample['y']]))
        if self.transform is not None:
            try:
                if self.logging:
                    print('transforming')
                    print(type(image))
                    print(image.shape)
                image = self.transform(image=image)['image']
            except Exception as e: 
                print('transforming failed')
                print(e)
                print(image_id)
        else:    
            image = transforms.ToTensor()(image) 
        house_features = torch.from_numpy(house_features.values)
        y = torch.from_numpy(np.array([y]))
        if self.logging:
            print(f'image: {image.shape}\nhouse: {house_features.shape}\nprice: {y.shape})')
        return image,house_features,y

#real_estate_combined_dataset = RealEstateCombinedDataset(csv_path='../dataset/df.csv',
#                                                         image_dir='../dataset/processed_images')
#real_estate_combined_dataloader = DataLoader(real_estate_combined_dataset, batch_size=32, shuffle=True)

class TwoInputsNet(pl.LightningModule):
    def __init__(self,df,image_dir,  lr: float = 1e-3, batch_size: int = 32,transform=None,
                 logging: bool = False, **kwargs):
        super(TwoInputsNet, self).__init__()
        self.conv = nn.Conv2d(3,8,kernel_size=3) 
        self.conv1 = nn.Conv2d(8,8,kernel_size=3)
        self.conv2 = nn.Conv2d(8,8,kernel_size=3) 
        self.fc1 = nn.Linear(3,3)
        self.fc2 = nn.Linear(26915,1024)
        self.fc3 = nn.Linear(1024,32) 
        self.fc4 = nn.Linear(32,1)
        self.df = df
        self.image_dir = image_dir
        self.lr = lr
        self.batch_size = batch_size
        self.transform = transform
        self.logging = logging


    def forward(self, input1, input2):
        if self.logging:
            print(
                f'input1 shape: {input1.shape}, input2 shape: {input2.shape}\n',
                f'input1 shape: {input1.dtype}, input2 shape: {input2.dtype}\n',
            )
        input1 = input1.to(torch.float32)
        input2 = input2.to(input1.dtype)
        # print(f'input2: {input2.shape}, {input2.dtype}')
        c = self.conv(input1)
        # print(f'conv(input1): {c.shape}, {c.dtype}')
        c = self.conv1(c)
        # print(f'conv1(input1): {c.shape}, {c.dtype}')
        c = F.relu(c)
        # print(f'F.relu(c): {c.shape}, {c.dtype}')
        c = self.conv2(c)
        # print(f'conv2(input1): {c.shape}, {c.dtype}')
        c = F.relu(c)
        # print(f'F.relu(c): {c.shape}, {c.dtype}')
        f = self.fc1(input2)
        # print(f'fc1(input2): {f.shape}, {f.dtype}')
        # now we can reshape `c` and `f` to 2D and concat them
        combined = torch.cat((c.view(c.size(0), -1),
                          f.view(f.size(0), -1)), dim=1)
        out = self.fc2(combined)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc4(out)
        return out

    def training_step(self, batch, batch_idx):
        image, tabular, y = batch

        criterion = torch.nn.L1Loss()
        # criterion = torch.nn.MSELoss(reduction='mean')
        y_pred = torch.flatten(self(image, tabular))
        y_pred = y_pred.double()
        y = torch.flatten(y)

        loss = criterion(y_pred, y)
        mape = mean_absolute_percentage_error(y_pred, torch.flatten(y))
        self.log('train_loss', loss)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss,"MAPE": mape, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        # print('Validation Step')
        image, tabular, y = batch
        y_pred = self.forward(image, tabular)
        criterion = torch.nn.L1Loss()
        # criterion = torch.nn.MSELoss(reduction='mean')
        y_pred = torch.flatten(self(image, tabular))
        y = torch.flatten(y)
        print(y_pred.shape, y.shape)
        y_pred = y_pred.double()

        val_loss = criterion(y_pred, y)
        self.log('train_loss', val_loss)

        mape = mean_absolute_percentage_error(y_pred, torch.flatten(y))
        print("MAPE", mape)
        #mape = val_loss
        return {"val_loss": val_loss, "MAPE": mape}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        image, tabular, y = batch
        y = torch.flatten(y)

        criterion = torch.nn.L1Loss()
        # criterion = torch.nn.MSELoss(reduction='mean')
        y_pred = torch.flatten(self(image, tabular))
        y_pred = y_pred.double()

        test_loss = criterion(y_pred, y)
        self.log('test_loss', test_loss)

        return {"test_loss": test_loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        logs = {"test_loss": avg_loss}
        return {"test_loss": avg_loss, "log": logs, "progress_bar": logs}    

    def setup(self, stage):
        house_dataset = RealEstateCombinedDataset(df=df, image_dir=self.image_dir, transform=self.transform)
        self.train_set, self.test_set,self.val_set = get_train_test(house_dataset, (0.6, 0.2))



    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=(self.lr))

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size,num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size,num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=8)


def experiment():
    pass



if __name__ == '__main__':
    df = preprocessing('../dataset/socal/socal2.csv')

    transform= A.Compose([
        A.Resize(64, 64),
        A.Normalize(),
        ToTensorV2()
    ])
    house_dataset = RealEstateCombinedDataset(df=df,
                                              image_dir='../dataset/socal/socal2/socal_pics',
                                              transform=transform)
    #check_dataset(house_dataset)
    train_set, test_set, val_set = get_train_test(house_dataset, (0.6, 0.2))
    print(type(train_set))
    # exit(0)
    model = TwoInputsNet(df=df,image_dir='../dataset/socal/socal2/socal_pics',transform=transform)

    # get_summary(model,house_dataset,batch_size=4)
    first_batch = next(iter(house_dataset))
    house_dataset_dataloader = DataLoader(house_dataset, batch_size=32, shuffle=True)
    batch =  next(iter(house_dataset_dataloader))
    A = batch[0]
    B = batch[1]
    C = batch[2]
    print(A.shape, B.shape, C.shape)
    stats = summary(model, input_data=(A,B),verbose=1)
    print(stats)
    from torchview import draw_graph

    model_graph = draw_graph(model,input_data=(A,B)) 
    print(model_graph.visual_graph)
    exit(0)
  
    #model = FineTunedModel(model_resnet50,torch.nn.L1Loss(),1e-3)  
    # early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=5000, patience=7, verbose=False, mode="min")
    # trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=10000, callbacks=[early_stop_callback])
    aim_logger = AimLogger(
        experiment='TwoInputsNet_Regression_3',
        train_metric_prefix='train_',
        val_metric_prefix='val_',
    )
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=3000,logger=aim_logger)
    #  PyTorch Lightning can help us with a learning rate finder. Through cyclically
    #  varying the learning rate with a few model restarts, we can find a reasonable starting learning rate.
    lr_finder = trainer.tuner.lr_find(model)
    fig = lr_finder.plot(suggest=True, show=True)
    new_lr = lr_finder.suggestion()
    model.hparams.lr = new_lr
   
    trainer.fit(model)
    trainer.test(model)
