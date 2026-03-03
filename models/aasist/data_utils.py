import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset

___author__ = "Hemlata Tak, Jee-weon Jung"
__email__ = "tak@eurecom.fr, jeeweon.jung@navercorp.com"


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x



# ---------------------------------------------------------
# 新增：专为 ASVspoof 5 设计的解析函数
# ---------------------------------------------------------
def genSpoof_list_asv5(dir_meta, is_train=False, is_eval=False):
    """
    解析 ASVspoof 5 的 .tsv 协议文件
    示例行: T_4850 T_0000000000 F - - - AC3 A05 spoof -
    """
    d_meta = {}
    file_list = []
    
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    for line in l_meta:
        cols = line.strip().split()
        
        # 兼容性检查：ASVspoof 5 标准协议应有不少于9列
        if len(cols) < 9:
            continue
            
        key = cols[1]          # 音频文件名，例如 T_0000000000
        attack_type = cols[7]  # 攻击类型，例如 A05。为后续 Phase 3 预留！
        label_str = cols[8]    # 标签：bonafide 或 spoof
        
        if is_eval:
            # 评估集可能没有给出 ground truth 标签
            file_list.append(key)
        else:
            file_list.append(key)
            # 兼容 Baseline：bonafide 为 1，spoof 为 0
            label_int = 1 if label_str == "bonafide" else 0
            
            # 为了方便后期 M-CAAS 扩展，我们将 value 存为字典 
            # (注意：在 Dataset __getitem__ 中我们会把它拆包)
            d_meta[key] = {
                "label": label_int,
                "attack_type": attack_type,
                "codec": cols[6]
            }

    if is_eval:
        return file_list
    return d_meta, file_list


# ---------------------------------------------------------
# 新增：专为 ASVspoof 5 设计的 Dataset
# ---------------------------------------------------------
class Dataset_ASVspoof5_train(Dataset):
    def __init__(self, list_IDs, labels_dict, base_dir):
        """
        list_IDs    : list of strings (音频文件名)
        labels_dict : 包含字典的字典 (由 genSpoof_list_asv5 生成)
        base_dir    : 数据集音频文件存放的根目录 (例如 'path/to/ASVspoof5/flac')
        """
        self.list_IDs = list_IDs
        self.labels_dict = labels_dict
        self.base_dir = Path(base_dir) 
        self.cut = 64600  # AASIST 默认输入长度

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        # 注意 ASVspoof5 的扩展名通常还是 .flac，如果是 .wav 请修改这里
        audio_path = self.base_dir / f"{key}.flac"
        
        X, _ = sf.read(str(audio_path))
        X_pad = pad_random(X, self.cut)
        x_inp = Tensor(X_pad)
        
        # 目前第一阶段，只向外暴露 0 或 1 作为标签，以兼容原版 main.py
        y = self.labels_dict[key]["label"]
        return x_inp, y


class Dataset_ASVspoof5_devNeval(Dataset):
    def __init__(self, list_IDs, base_dir):
        self.list_IDs = list_IDs
        self.base_dir = Path(base_dir)
        self.cut = 64600

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        audio_path = self.base_dir / f"{key}.flac"
        
        X, _ = sf.read(str(audio_path))
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        
        return x_inp, key