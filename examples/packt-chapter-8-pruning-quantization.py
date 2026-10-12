# Фрагмент официального кода Packt: Mastering Transformers
# MIT License, Copyright (c) 2021 Packt
# Источник: https://github.com/PacktPublishing/Mastering-Transformers/blob/f088d75fb24f583a581a5a99fa02ccf445b77dfc/CH08/CH08a_Pruning_Quantization.ipynb
import torch
from torch.nn.utils import prune
from sentence_transformers import SentenceTransformer

distilroberta = SentenceTransformer("stsb-distilroberta-base-v2")
pruner = prune.L1Unstructured(amount=0.2)

state_dict = distilroberta.state_dict()
for key in state_dict.keys():
    if "weight" in key:
        state_dict[key] = pruner.prune(state_dict[key])
distilroberta.load_state_dict(state_dict)

distilroberta = torch.quantization.quantize_dynamic(
    model=distilroberta,
    qconfig_spec={torch.nn.Linear: torch.quantization.default_dynamic_qconfig},
    dtype=torch.qint8,
)
