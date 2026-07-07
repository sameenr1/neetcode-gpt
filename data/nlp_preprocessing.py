import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        all_sentences = []
        vocab = set()
        for i in range (len(positive)):
            for word in positive[i].split():
                vocab.add(word)

            for word in negative[i].split():
                vocab.add(word)

        sorted_vocab = sorted(vocab)

        word_to_id = {}
        id = 1

        for word in sorted_vocab:
            word_to_id[word] = id
            id += 1
        
        max_len = -1
        for i in range (len(positive)):
            if len(positive[i].split()) >= max_len:
                max_len = len(positive[i].split())

            if len(negative[i].split()) >= max_len:
                max_len = len(negative[i].split())

        output = []

        for sentence in positive:
            list = []
            for word in sentence.split():
                list.append(word_to_id[word])

            while (len(list) < max_len):
                list.append(0)

            output.append(list)

           
        for sentence in negative:
            list = []
            for word in sentence.split():
                list.append(word_to_id[word])

            while (len(list) < max_len):
                list.append(0)

            output.append(list)

        output_tensor = torch.tensor(output)

        return output_tensor
        

