import collections
import unicodedata
import os
import re

def character_to_idx():
    char_to_idx = {}

    for i in range(256):
        char_to_idx[chr(i)] = i

    return char_to_idx

class Tokenizer:
    def __init__(self, num_merges, oov_token=None):
        self.num_merges = num_merges
        self.oov_token = oov_token
        self.word_encode = character_to_idx()
        self.word_decode = {idx:char for char, idx in self.word_encode.items()}
        self.merges_idx = {}
        self.merges_char = {}
        self.pattern = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d|n't|\s?[a-zA-Z]+|\s?[^\x00-\x7F]+|\s?\d+|\s?[^\w\s]+")

    def prepare_ids(self, text):
        return [list(word.encode('utf-8')) for word in text]

    def get_pair(self, chunked_ids, stats=None):
        pairs = collections.defaultdict(int) if stats is None else stats

        for i in range(len(chunked_ids) - 1):
            pairs[chunked_ids[i], chunked_ids[i + 1]] = pairs.get((chunked_ids[i], chunked_ids[i + 1]), 0) + 1

        return pairs
    
    def merge_pair(self, ids_dataset, pair, idx):
        new_ids = []

        for ids in ids_dataset:
            new_ids.append([])
            for chunked_ids in ids:
                new_chunked_ids = []
                i = 0
                while i < len(chunked_ids):
                    if i < len(chunked_ids)-1 and (chunked_ids[i], chunked_ids[i + 1]) == pair:
                        new_chunked_ids.append(idx)
                        i += 2
                    else:
                        new_chunked_ids.append(chunked_ids[i])
                        i += 1
                new_ids[-1].append(new_chunked_ids)
        
        return new_ids


    def fit(self, dataset, verbose=True):
        chunked_dataset = [re.findall(self.pattern, text) for text in dataset]
        ids_dataset = [self.prepare_ids(text) for text in chunked_dataset]

        for i in range(self.num_merges):
            pairs = {}
            for ids in ids_dataset: 
                for chunked_ids in ids: pairs = self.get_pair(chunked_ids, pairs)
            best = max(pairs, key=pairs.get)
            if pairs[best] < 2: break
            self.merges_idx[best] = 256 + i
            self.word_decode[256+i] = self.word_decode[best[0]] + self.word_decode[best[1]]
            self.merges_char[self.word_decode[best[0]], self.word_decode[best[1]]] = self.word_decode[256+i]
            if verbose: print(f"Num merge {i+1}/{self.num_merges}: ({self.word_decode[best[0]]}, {self.word_decode[best[1]]}) -> '{self.word_decode[256+i]}'")
            ids_dataset = self.merge_pair(ids_dataset, best, 256 + i)
            

    def encode(self, text):
        encoded = []
        words = re.findall(self.pattern, text)
        ids = self.prepare_ids(words)

        for chunked_ids in ids:
            while True:
                pairs = self.get_pair(chunked_ids)
                if not pairs: break
                else:
                    best = min(pairs, key=lambda p: self.merges_idx.get(p, float('inf')))
                    if best not in self.merges_idx: break
                    chunked_ids = self.merge_pair([[chunked_ids]], best, self.merges_idx[best])[0][0]
            encoded.extend(chunked_ids)

        return encoded

    def decode(self, ids):
        text = ''.join([self.word_decode[token] for token in ids])

        return text
    
    def save(self, merge_path, vocab_path):
        if not os.path.exists(merge_path): os.mkdir(merge_path)
        if not os.path.exists(vocab_path): os.mkdir(vocab_path)
        with open(os.path.join(merge_path, 'merges.txt'), 'w') as f:
            for (idx1, idx2) in self.merges_idx:
                f.write(f"{idx1} {idx2}\n")

        with open(os.path.join(vocab_path, 'vocab.txt'), 'w') as f:
            for idx, word in self.word_decode.items():
                word = word.replace("\n", "[ENL]")
                f.write(f'{idx}->{word}\n')
    
    def load(self, merge_path, vocab_path):
        with open(merge_path, 'r') as f:
            for i, line in enumerate(f):
                idx1, idx2 = line.strip().split()
                self.merges_idx[(int(idx1), int(idx2))] = 256 + i

        with open(vocab_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                idx, word = (line.strip().split("->"))
                if i in range(0,256): continue
                else:
                    word = word.replace("[ENL]", "\n")
                    self.word_decode[i] = word

        