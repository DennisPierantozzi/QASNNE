import os
import glob

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as transforms
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import json
import torch

class PitVQASentence(Dataset):
    def __init__(self, seq, folder_head, folder_tail):

        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
	        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # files, question and answers
        filenames = []
        for curr_seq in seq:
            filenames = filenames + glob.glob(folder_head + str(curr_seq) + folder_tail)
        self.vqas = []
        for file in filenames:
            file_data = open(file, "r")
            lines = [line.strip("\n") for line in file_data if line != "\n"]
            file_data.close()
            for line in lines:
                self.vqas.append([file, line])
        print('Total files: %d | Total question: %.d' % (len(filenames), len(self.vqas)))

    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):
        qa_full_path = Path(self.vqas[idx][0])
        video_dir = qa_full_path.parent.parent  # .../PIT_Dataset/video_xx
        frame_stem = qa_full_path.stem 

        # img_path = seq_path / 'images' / video_folder
        # file_name = self.vqas[idx][0].split('/')[-1] 

        # img
        # img_loc = os.path.join(img_path, file_name.split('.')[0] + '.png')
        img_loc = video_dir / f"{frame_stem}.png"
        raw_image = Image.open(img_loc).convert('RGB')
        img = self.transform(raw_image)

        # question and answer
        question, answer = self.vqas[idx][1].split('|')
        return img, question, answer, str(img_loc)



class EndoVis18VQA(Dataset):
    def __init__(self, seq, folder_head, folder_tail):

        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # files, question and answers
        filenames = []
        for curr_seq in seq:
            filenames = filenames + glob.glob(folder_head + str(curr_seq) + folder_tail)
        self.vqas = []
        for file in filenames:
            file_data = open(file, "r")
            lines = [line.strip("\n") for line in file_data if line != "\n"]
            file_data.close()
            for line in lines:
                self.vqas.append([file, line])
        print('Total files: %d | Total question: %.d' % (len(filenames), len(self.vqas)))

    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):
        qa_full_path = Path(self.vqas[idx][0])
        seq_path = qa_full_path.parents[2]
        file_name = self.vqas[idx][0].split('/')[-1] 

        # img
        img_loc = os.path.join(seq_path, 'left_fr', file_name.split('_')[0] + '.png')
        raw_image = Image.open(img_loc).convert('RGB')
        img = self.transform(raw_image)

        # question and answer
        question, answer = self.vqas[idx][1].split('|')
        return img, question, answer, str(img_loc)
    