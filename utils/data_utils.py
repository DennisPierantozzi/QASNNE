from dataloader import EndoVis18VQA, PitVQASentence
from torch.utils.data import DataLoader, Subset
import torch


def build_dataloader(dataset_name: str, batch_size: int=4):
    if dataset_name == "Endovis18VQA_new_template":
        #val_seq = ['2','6','12']
        val_seq = ['1','5','16']
        folder_head = r'/SAN/medic/Cholec/Endovis-OOT-GPT5/seq_'
        folder_tail = '/vqa/Sentence/*.txt'
        ds = EndoVis18VQA(val_seq, folder_head, folder_tail)
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    elif dataset_name == "Endovis18VQA_old_template":
        #val_seq = ['2','6','12']
        val_seq = ['1','5','16']
        folder_head = r'/SAN/medic/Cholec/EndoVis-18-VQA/seq_'
        folder_tail = '/vqa/Sentence/*.txt'
        ds = EndoVis18VQA(val_seq, folder_head, folder_tail)
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    
    elif dataset_name == "PitVQASentence":
        val_seq = ['02', '06', '12']
        folder_head = r'/SAN/medic/Cholec/PitSentence/PIT_Dataset/video_'
        folder_tail = '/qa/*.txt'

        ds = PitVQASentence(val_seq, folder_head, folder_tail)

        # 15% as validation only
        keep = int(len(ds) * 0.05)
        g = torch.Generator().manual_seed(42)          
        indices = torch.randperm(len(ds), generator=g)[:keep]
        ds = Subset(ds, indices)

        print(f"Using {keep} / {len(ds)} samples for PitVQASentence subset")

        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")