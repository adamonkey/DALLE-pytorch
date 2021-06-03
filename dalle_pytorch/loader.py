from pathlib import Path
from random import randint, choice

import PIL

from torch.utils.data import Dataset
from torchvision import transforms as T

import csv
import boto3
from boto3.s3.transfer import S3Transfer
import os
import shutil
import json
import pandas as pd 

session = boto3.Session(profile_name='default')
client = session.client('s3')
transfer = S3Transfer(client)

class TextImageDataset(Dataset):
    def __init__(self,
                 folder,
                 text_len=256,
                 image_size=128,
                 truncate_captions=False,
                 resize_ratio=0.75,
                 tokenizer=None,
                 shuffle=False
                 ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()
        self.shuffle = shuffle
        path = Path(folder)
        self.image_path = f'{path}/images'
        
        if not os.path.exists(self.image_path):
            os.mkdir(self.image_path)
            
        df = pd.read_csv("Train_GCC-dalle.tsv", sep='\t')
        print('Loading training data finished.')

        self.text_files = json.loads(df.set_index('filename').to_json(orient='index'))
        print('Creating text_files finished.')
        self.keys = list(self.text_files.keys())
        
        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.resize_ratio = resize_ratio
        self.tokenizer = tokenizer
        self.image_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB')
            if img.mode != 'RGB' else img),
            T.RandomResizedCrop(image_size,
                                scale=(self.resize_ratio, 1.),
                                ratio=(1., 1.)),
            T.ToTensor()
        ]) # T: module from torchvision 

    def __len__(self):
        return len(self.keys)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        key = self.keys[ind]

        # LOAD TEXT
        text_file = self.text_files[key]
        descriptions = text_file['caption'].split('\n') # shouldn't need this... interesting!
        descriptions = list(filter(lambda t: len(t) > 0, descriptions))
        try:
            description = choice(descriptions)
        except IndexError as zero_captions_in_file_ex:
            print(f"An exception occurred trying to load file {text_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        tokenized_text = self.tokenizer.tokenize(
            description,
            self.text_len,
            truncate_text=self.truncate_captions
        ).squeeze(0)

        # LOAD IMAGE
        s3_image_path = f's3illustration/{key}'
        local_image_path = f'{self.image_path}/{key}'
        try:
            if not os.path.exists(local_image_path):
                transfer.download_file(s3_image_path, local_image_path)
            image_tensor = self.image_transform(PIL.Image.open(local_image_path))
        except (PIL.UnidentifiedImageError, OSError) as corrupt_image_exceptions:
            print(f"An exception occurred trying to load file {local_image_path}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        # Success
        return tokenized_text, image_tensor

    def flush():
        '''ocassionally removes all images in self.image_path'''
        shutil.rmtree(self.image_dir)
