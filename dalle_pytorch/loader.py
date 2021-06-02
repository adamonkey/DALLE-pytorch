from pathlib import Path
from random import randint, choice

import PIL

from torch.utils.data import Dataset
from torchvision import transforms as T

import csv
import boto3
from boto3.s3.transfer import S3Transfer
import os

session = boto3.Session(profile_name='default')
client = session.client('s3')
transfer = S3Transfer(client)

# TODO: need to write the function to deal with the .tsv dataset

'''
img = df.iloc[42][1]
(zlib.crc32(img.encode('utf-8')) & 0xffffffff)
'''

import pandas as pd
import zlib
def create_captions_folder(folder):
    
    assert folder
    
    df = pd.read_csv('Train_GCC-training.tsv', sep='\t', header=None)
    df.columns=['caption','url']
    for i, row in df.iterrows():
        unique = (zlib.crc32(row['caption'].encode('utf-8')) & 0xffffffff)
        fname = f'{i}_{unique}'
        
        with open(fname, 'wb') as out_file:
            out_file.write(row['caption'])


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


        with open("Train_GCC-training.tsv") as fd:
            rd = csv.reader(fd, delimiter="\t", quotechar='"')
            for row in rd:
                print(row)
                
        text_files = [*path.glob('**/*.txt')]
        self.text_files = {text_file.stem: text_file for text_file in text_files}

        # original code takes intersection, this is also unnecessary...

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

        # OPTIONS: 
        # - store it in memory
        # - (DO THIS ONE) actually write the text files. interesting ... might make sense...if I can do it. (now I just have touch...?)

        # LOAD TEXT
        text_file = self.text_files[key]
        descriptions = text_file.read_text().split('\n') # shouldn't need this... interesting!
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
        s3_image_path = f's3illustration'
        local_image_path = f'{self.image_folder}'
        try:
            if not os.path.exists(local_image_path):
                
                transfer.download_file(s3_image_path, local_image_path)
            # put S3 loading here too ... 
            image_tensor = self.image_transform(PIL.Image.open(local_image_path))
        except (PIL.UnidentifiedImageError, OSError) as corrupt_image_exceptions:
            print(f"An exception occurred trying to load file {local_image_path}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        # Success
        return tokenized_text, image_tensor

    def flush():
      '''remove all images in self.image_dir'''
      # TODO
      # occasionally flush all the images ... (on start for example)
      # and like ... errr ... 


# this is kind of unnecessary. the text_file part ... 
# LOAD TEXT (do I store it in memory...? that's kind of ... hmm ... it's 1GB in memory)
# that's too much...
# well ... actually without the links, it might be half. 
# I think I have to store it in memory

