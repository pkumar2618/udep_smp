from allennlp.data.iterators import DataIterator
from tqdm import tqdm
from scipy.special import expit  # the sigmoid function
from allennlp.models import Model
from typing import Iterable
from allennlp.data import Instance
import torch
import numpy as np
from allennlp.nn import util as nn_util
from dl_lib.dl_utilities import merge_dict
import json

def tonp(tsr): return tsr.detach().cpu().numpy()
#USE_GPU = torch.cuda.is_available()
class Predictor:
    def __init__(self, model: Model, iterator: DataIterator,
                 cuda_device: int = -1) -> None:
        self.model = model
        self.iterator = iterator
        self.cuda_device = cuda_device

    def _extract_data(self, batch):
        out_dict = self.model(**batch)
        # return expit(tonp(out_dict["sentence_spo_logits"]))
        return out_dict["sentence_spo_logits"]

    def predict(self, ds: Iterable[Instance], write_pred = False):
        pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
        self.model.eval()
        pred_generator_tqdm = tqdm(pred_generator,
                                   total=self.iterator.get_num_batches(ds))
        json_list = []
        # json_items_dict = {"sentence_candidate": None, "score": None}
        with torch.no_grad():
            instance_count = 0
            for batch_sentence_candidates in pred_generator_tqdm:
                batch_input = nn_util.move_to_device(batch_sentence_candidates, self.cuda_device)
                batch_scores = self._extract_data(batch_input)
                for sentence_candidate, sentence_spo_dict, score in zip(ds[instance_count]["sentence_spo"], ds[instance_count]["sentence_spo_raw"], batch_scores):
                    input_dict = sentence_spo_dict
                    output_dict = {"sentence_candidate": " ".join([x.text for x in sentence_candidate]), "cross_emb_score": score.item()}
                    json_list.append(merge_dict(input_dict, output_dict))

                instance_count += 1

        if write_pred:
            with open('output_prediction.json', 'w') as f_write:
                json.dump(json_list, f_write, indent=4)
        else:
            return json_list

        # return np.concatenate(json_list, axis=0)
