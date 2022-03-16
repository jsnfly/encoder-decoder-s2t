import torch
from torch.utils.data import Dataset
from pathlib import Path


class Wav2VecFeaturesDataset(Dataset):
    def __init__(self, data_path, prompt):
        self.path = Path(data_path)
        self.files = list(self.path.iterdir())
        self.prompt = prompt + ' '

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        eg = torch.load(file_path)
        eg['sentence'] = self.prompt + eg['sentence']
        eg['file_path'] = file_path
        return eg


# TODO: Somehow masks do not work yet (bad performace), but Training also works w/o using the mask.
def make_collate_fn(tokenizer):
    def collate_fn(examples):
        wav2vec_feats = [eg['wave2vec_features'] for eg in examples]
        max_len = len(max(wav2vec_feats, key=len))
        padded_feats, attention_masks = [], []
        for feats in wav2vec_feats:
            num_pads = max_len - len(feats)
            padded_feats.append(torch.cat([feats, torch.zeros((num_pads, feats.shape[-1]), device=feats.device)]))
            if num_pads > 0:
                mask = torch.zeros((max_len,), device=feats.device).long()
                mask[:-num_pads] = 1
            else:
                mask = torch.ones((max_len,), device=feats.device).long()
            attention_masks.append(mask)

        encoder_hidden_states = torch.stack(padded_feats, dim=0)
        encoder_attention_masks = torch.stack(attention_masks, dim=0).bool()
        input_ids = tokenizer([eg['sentence'] for eg in examples],
                              return_tensors='pt', padding=True, add_special_tokens=False).input_ids
        return encoder_hidden_states, encoder_attention_masks, input_ids
    return collate_fn