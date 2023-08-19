
from torch.utils.data import Dataset, DataLoader,random_split
import numpy as np
import torch 
import os
from PIL import Image
from torchvision import transforms, utils
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchmetrics.functional import mean_absolute_percentage_error
import random
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
"""
 #   Column                       Non-Null Count  Dtype  
---  ------                       --------------  -----  
 0   id_annonce                   37368 non-null  int64  
 1   property_type                37368 non-null  object 
 2   approximate_latitude         37368 non-null  float64
 3   approximate_longitude        37368 non-null  float64
 4   city                         37368 non-null  object 
 5   postal_code                  37368 non-null  int64  
 6   size                         36856 non-null  float64
 7   floor                        9743 non-null   float64
 8   land_size                    15581 non-null  float64
 9   energy_performance_value     19068 non-null  float64
 10  energy_performance_category  19068 non-null  object 
 11  ghg_value                    18530 non-null  float64
 12  ghg_category                 18530 non-null  object 
 13  exposition                   9094 non-null   object 
 14  nb_rooms                     35802 non-null  float64
 15  nb_bedrooms                  34635 non-null  float64
 16  nb_bathrooms                 24095 non-null  float64
 17  nb_parking_places            37368 non-null  float64
 18  nb_boxes                     37368 non-null  float64
 19  nb_photos                    37368 non-null  float64
 20  has_a_balcony                37368 non-null  float64
 21  nb_terraces                  37368 non-null  float64
 22  has_a_cellar                 37368 non-null  float64
 23  has_a_garage                 37368 non-null  float64
 24  has_air_conditioning         37368 non-null  float64
 25  last_floor                   37368 non-null  float64
 26  upper_floors                 37368 non-null  float64
 27  price                        37368 non-null  float64
dtypes: float64(21), int64(2), object(5)
"""


def lat_long_to_cartesian(long,lat):
    x = np.cos(lat) * np.cos(long)
    y = np.cos(lat) * np.sin(long)
    z = np.sin(lat)
    return x,y,z 

def preprocessing(df):
    df = df[['id_annonce','approximate_latitude','approximate_longitude','price','size','nb_rooms',
    'nb_bedrooms','nb_bathrooms','nb_parking_places','nb_boxes','nb_photos',
    'last_floor','upper_floors']].copy()
    print('preprocessing')
    print(df.columns)
    print(df.info())
    # Get all numerical columns
    df = df.select_dtypes(include=[np.number])    
    print(df.head())

    # df = df.drop(columns=['street', 'citi','n_citi'],axis=0)
    df = df.dropna()
    df = df.reset_index(drop=True)
    df['price'] = df['price'].astype(np.double)
    # standerdize(df,['sqft','price'])
    # normalize(df,['sqft','price'])
    # df.to_csv('../dataset/df.csv', index=False)
    print(f'data shape: {df.shape}')
    print(df.info)
    print('preprocessing done')
    return df


def create_df(dataset_path,dataset_path2):
    df = pd.read_csv(dataset_path)
    df2 = pd.read_csv(dataset_path2)
    df = pd.merge(df,df2,on='id_annonce',how='inner')
    return df



class RealEstateILBDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        print(
            f"Creating dataset with {len(dataframe)} samples and {len(dataframe['id_annonce'].unique())} unique ads"
        )
        print(dataframe.info())
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        images = list()
        ann_id = int(row['id_annonce'])
        y = row['price']
        image_dir = os.path.join(self.root_dir, f"ann_{ann_id}/")
        image_files = os.listdir(image_dir)
        if len(image_files) < 6:
            print(f'Not enough images for ad {ann_id} {len(image_files)}')
            # choose random images from the same directory to complete the 6 images
            while len(image_files) < 6:
                image_files += random.sample(image_files,1 )
        for image_file in image_files:
            image = Image.open(os.path.join(image_dir, image_file))
            image = np.array(image).astype(np.float32)
            if self.transform:
                print('transforming')
                print(type(image))
                image = self.transform(image=image)
                print(type(image['image']))
            images.append(image['image'])
        print(type(images[0]))    
        images = torch.stack(images)
        print("row.values",type(row.values))
        print("row.values",type(row))
        print("row.values",row.values)
        data = torch.from_numpy(row.values).float()
        return images, data,y


class RealEstateILB1Dataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        print(
            f"Creating dataset with {len(dataframe)} samples and {len(dataframe['id_annonce'].unique())} unique ads"
        )
        print(dataframe.info())
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        images = list()
        ann_id = int(row['id_annonce'])
        y = row['price']
        image_dir = os.path.join(self.root_dir, f"ann_{ann_id}/")
        image_files = os.listdir(image_dir)
        if len(image_files) < 6:
            print(f'Not enough images for ad {ann_id} {len(image_files)}')
            # choose random images from the same directory to complete the 6 images
            while len(image_files) < 6:
                image_files += random.sample(image_files,1 )
        for image_file in image_files:
            image = Image.open(os.path.join(image_dir, image_file))
            image = np.array(image).astype(np.float32)
            if self.transform:
                print('transforming')
                print(type(image))
                image = self.transform(image=image)
                print(type(image['image']))
            images.append(image['image'])
        print(type(images[0]))    
        images = torch.stack(images)
        print("row.values",type(row.values))
        print("row.values",type(row))
        print("row.values",row.values)
        data = torch.from_numpy(row.values).float()
        return images, data,y




def check_dataset(dataset):
    for i in range(10):
        image,house_features,y = dataset[i]
        print(i, image.shape, house_features.shape, y.shape)



if __name__ ==  '__main__':
    df = create_df('../../dataset/ILB/X_train_J01Z4CN.csv','../../dataset/ILB/y_train_OXxrJt1.csv') 
    df = preprocessing(df)

    
    transform = A.Compose([
                A.Resize(height=64, width=64),
                A.RandomBrightness(),
                A.RandomContrast(),
                ToTensorV2()])
    transform = A.Compose([
        A.Resize(64, 64),
        A.Normalize(),
        ToTensorV2()
        ])
    # house_dataset = RealEstateILBDataset(df,train=True,
    #                                           image_dir='../dataset/ILB/reduced_images',
    #                                           transform=transform)
  

    house_dataset = RealEstateILBDataset(df,root_dir='../../dataset/ILB/reduced_images/train',
                                              transform=transform)
  
    
    check_dataset(house_dataset)