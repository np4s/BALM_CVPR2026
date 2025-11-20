import torch
import math
import random
import pickle
from tqdm import tqdm
from utils.prepare_idt import generate_modal_mask

def load_iemocap(path, ratio=0.1, modal_MR=None):
    with open(path, 'rb') as f:
        unsplit = pickle.load(f)
        
    speaker_to_idx = {"M": 0, "F": 1}

    data = {
        "train": [], "dev": [], "test": [],
    }
    testVid = list(unsplit["testVid"])

    trainVid = list(unsplit["trainVid"])
    random.shuffle(trainVid)
    dev_size = int(len(trainVid) * ratio)
    spliter = {
        "train": trainVid[dev_size:],
        "dev": trainVid[:dev_size],
        "test": testVid
    }
                
    if modal_MR is not None:
        modal_mask = {}
        for split in spliter:
            lengths = [len(unsplit['label'][uid]) for uid in spliter[split]]
            modal_mask[split] = generate_modal_mask(modal_MR, lengths, spliter[split])
    else:
        modal_mask = {"train": None, "dev": None, "test": None}

    for split in data:
        for uid in tqdm((spliter[split]), desc=split):
            text = unsplit["text"][uid]
            audio = unsplit["audio"][uid]
            visual = unsplit["visual"][uid]

            split_mask = modal_mask[split]
            if split_mask is not None:
                text = [mask*feat for mask, feat in zip(split_mask["t"][uid], text)]
                audio = [mask*feat for mask, feat in zip(split_mask["a"][uid], audio)]
                visual = [mask*feat for mask, feat in zip(split_mask["v"][uid], visual)]
            
            data[split].append(
                {
                    "uid": uid,
                    "speakers": [speaker_to_idx[speaker] for speaker in unsplit["speaker"][uid]],
                    "labels": unsplit["label"][uid],
                    "text": text,
                    "audio": audio,
                    "visual": visual,
                    "speaker_mask": [[1, 0] if speaker == 'M' else [0, 1] for speaker in unsplit["speaker"][uid]],
                    "sentence": unsplit["sentence"][uid],
                    "modal_mask": None if split_mask is None else {
                        "text": split_mask["t"][uid],
                        "audio": split_mask["a"][uid],
                        "visual": split_mask["v"][uid],
                    }
                }
            )
    return data

def load_mosei(path, ratio=0.1, modal_MR=None):
    with open(path, "rb") as f:
        unsplit = pickle.load(f)

    data = {
        "train": [], "dev": [], "test": [],
    }
    testVid = list(unsplit["testVid"])
    devVid = list(unsplit["valVid"])
    trainVid = list(unsplit["trainVid"])
    random.shuffle(trainVid)
    spliter = {
        "train": trainVid,
        "dev": devVid,
        "test": testVid
    }
                
    if modal_MR is not None:
        modal_mask = {}
        for split in spliter:
            lengths = [len(unsplit['label'][uid]) for uid in spliter[split]]
            modal_mask[split] = generate_modal_mask(modal_MR, lengths, spliter[split])
    else:
        modal_mask = {"train": None, "dev": None, "test": None}

    for split in data:
        for uid in tqdm((spliter[split]), desc=split):
            text = unsplit["text"][uid]
            audio = unsplit["audio"][uid]
            visual = unsplit["visual"][uid]

            split_mask = modal_mask[split]
            if split_mask is not None:
                text = [mask*feat for mask, feat in zip(split_mask["t"][uid], text)]
                audio = [mask*feat for mask, feat in zip(split_mask["a"][uid], audio)]
                visual = [mask*feat for mask, feat in zip(split_mask["v"][uid], visual)]
                
            data[split].append(
                {
                    "uid": uid,
                    "speakers": [0]*len(unsplit["speaker"][uid]),
                    "labels": unsplit["label"][uid],
                    "text": text,
                    "audio": audio,
                    "visual": visual,
                    "speaker_mask": [[1]]*len(unsplit["speaker"][uid]),
                    "sentence": unsplit["sentence"][uid],
                    "modal_mask": None if split_mask is None else {
                        "text": split_mask["t"][uid],
                        "audio": split_mask["a"][uid],
                        "visual": split_mask["v"][uid],
                    }
                }
            )

    return data

class Dataloader:
    def __init__(self, data, args):
        self.data = data
        self.batch_size = args.batch_size
        self.num_batches = math.ceil(len(data) / self.batch_size)
        self.dataset = args.dataset
        self.num_speakers = args.dataset_speaker_dict[self.dataset]
        self.embedding_dim = args.embedding_dim[self.dataset]

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        batch = self.raw_batch(index)
        return self.padding(batch)

    def raw_batch(self, index):
        assert index < self.num_batches, "batch_idx %d > %d" % (
            index, self.num_batches)
        batch = self.data[index *
                          self.batch_size: (index + 1) * self.batch_size]
        return batch

    def padding(self, samples):
        batch_size = len(samples)
        text_len_tensor = torch.tensor(
            [len(s["text"]) for s in samples]).long()
        uid = [s["uid"] for s in samples]
        mx = torch.max(text_len_tensor).item()
        have_masked = samples[0]['modal_mask'] is not None

        audio_tensor = torch.zeros((batch_size, mx, self.embedding_dim['a']))
        text_tensor = torch.zeros((batch_size, mx, self.embedding_dim['l']))
        visual_tensor = torch.zeros((batch_size, mx, self.embedding_dim['v']))
        speaker_tensor = torch.zeros((batch_size, mx)).long()
        speaker_mask_tensor = torch.zeros((batch_size, mx, self.num_speakers))

        labels = []
        utterances = []
        
        text_mask_tensor = torch.zeros((batch_size, mx, 1)).long()
        audio_mask_tensor = torch.zeros((batch_size, mx, 1)).long()
        visual_mask_tensor = torch.zeros((batch_size, mx, 1)).long()
            
        for i, s in enumerate(samples):
            cur_len = len(s["text"])
            utterances.append(s["sentence"])

            tmp_t = [torch.tensor(t) for t in s["text"]]
            tmp_a = [torch.tensor(a) for a in s["audio"]]
            tmp_v = [torch.tensor(v) for v in s["visual"]]
            tmp_mspk = [torch.tensor(mspk) for mspk in s["speaker_mask"]]

            tmp_t = torch.stack(tmp_t)
            tmp_a = torch.stack(tmp_a)
            tmp_v = torch.stack(tmp_v)
            tmp_mspk = torch.stack(tmp_mspk)

            text_tensor[i, :cur_len, :] = tmp_t
            audio_tensor[i, :cur_len, :] = tmp_a
            visual_tensor[i, :cur_len, :] = tmp_v
            speaker_mask_tensor[i, :cur_len, :] = tmp_mspk
            speaker_tensor[i, :cur_len] = torch.tensor(s["speakers"])

            labels.extend(s["labels"])
            
            if have_masked:                
                text_mask_tensor[i, :cur_len] = torch.tensor(s["modal_mask"]["text"]).unsqueeze(-1).long()
                audio_mask_tensor[i, :cur_len] = torch.tensor(s["modal_mask"]["audio"]).unsqueeze(-1).long()
                visual_mask_tensor[i, :cur_len] = torch.tensor(s["modal_mask"]["visual"]).unsqueeze(-1).long()
            else:
                text_mask_tensor[i, :cur_len] = torch.ones((cur_len, 1)).long()
                audio_mask_tensor[i, :cur_len] = torch.ones((cur_len, 1)).long()
                visual_mask_tensor[i, :cur_len] = torch.ones((cur_len, 1)).long()

        label_tensor = torch.tensor(labels)
        if not 'mosei' == self.dataset:
            label_tensor = label_tensor.long()

        data = {
            "uid": uid,
            "length": text_len_tensor,
            "tensor": {
                "t": text_tensor,
                "a": audio_tensor,
                "v": visual_tensor,
            },
            "speaker_tensor": speaker_tensor,
            "label_tensor": label_tensor,
            "speaker_mask_tensor": speaker_mask_tensor,
            "utterance_texts": utterances,
            "modal_mask_tensor":{
                "t": text_mask_tensor,
                "a": audio_mask_tensor,
                "v": visual_mask_tensor,
            }
        }

        return data

    def shuffle(self):
        random.shuffle(self.data)