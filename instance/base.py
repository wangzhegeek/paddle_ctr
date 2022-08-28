
import os
import abc
import numpy as np
import hashlib

import paddle.fluid as fluid


class Int8Hash:
    BYTES = 8
    BITS = BYTES * 8
    BITS_MINUS1 = BITS - 1
    MIN = -(2**BITS_MINUS1)
    MAX = 2**BITS_MINUS1 - 1

    @classmethod
    def as_dict(cls, texts):
        return {cls.as_int(text): text for text in texts}  # Intentionally reversed.

    @classmethod
    def as_int(cls, text):
        seed = text.encode()
        hash_digest = hashlib.shake_128(seed).digest(cls.BYTES)
        hash_int = int.from_bytes(hash_digest, byteorder='big', signed=True)
        assert cls.MIN <= hash_int <= cls.MAX
        return hash_int

    @classmethod
    def as_list(cls, texts):
        return [cls.as_int(text) for text in texts]


class ReaderBase(object):
    def __init__(self, places, sparse_slots, dense_slots, seq_slots=[], batch_szie=128, hash_size=100000, capacity=64, sep="\t", iterable=False):
        self.places = places
        self.sparse_slots = sparse_slots
        self.dense_slots = dense_slots
        self.seq_slots = seq_slots
        self.hash_size = hash_size
        self.total_slots = dense_slots + sparse_slots + seq_slots
        self.slot2index = dict()
        self.default = "0"
        self.sep = sep
        for i, slot in enumerate(self.total_slots):
            self.slot2index[slot] = i + 1
        self.label = None
        self.dense_var = []
        self.sparse_var = []
        self.seq_var = []
        self.dense_var_map = {}
        self.sparse_var_map = {}
        self.seq_var_map = {}
        self.data_loader = None
        self.iterable = iterable
        self.batch_size = batch_szie
        self.capacity = capacity
        self.data_var = []

    def generate_sample(self, l):
        def reader():
            line = l.strip().split(self.sep)
            label = float(line[0].split(':')[1])
            output = [("label", [label])] + [(i, []) for i in self.total_slots]
            for i in line[1:]:
                slot, feasign_str = i.split(':')
                feasign_str = self.default if feasign_str == "" else feasign_str
                if slot in self.sparse_slots or slot in self.seq_slots:
                    # print(str(self.sparse_slots.index(slot)) + feasign_str)
                    feasign = Int8Hash.as_int(str(self.sparse_slots.index(slot)) + feasign_str)%self.hash_size
                    # print(feasign)
                elif slot in self.dense_slots:
                    feasign = float(feasign_str)
                else:
                    continue
                output[self.slot2index[slot]][1].append(feasign)
            yield output
        return reader

    def get_reader(self, data_path):
        def gen_reader():
            with open(data_path, 'r') as f:
                for line in f:
                    iter = self.generate_sample(line)
                    for parsed_line in iter():
                        values = []
                        for pased in parsed_line:
                            values.append(pased[1])
                        yield values
                        # print(values[-1])
        return gen_reader

    def get_input_var(self):
        self.label = fluid.layers.data(name="label", shape=[1], dtype="float32")
        self.data_var.append(self.label)
        for i, slot in enumerate(self.dense_slots):
            data = fluid.layers.data(name=slot, shape=[1], dtype="float32")
            self.dense_var.append(data)
            self.dense_var_map[slot] = data
            self.data_var.append(data)
        for i, slot in enumerate(self.sparse_slots):
            data = fluid.layers.data(name=slot, shape=[1], lod_level=1, dtype="int64")
            self.sparse_var.append(data)
            self.sparse_var_map[slot] = data
            self.data_var.append(data)
        for i, slot in enumerate(self.seq_slots):
            data = fluid.layers.data(name=slot, shape=[1], lod_level=2, dtype="int64")
            self.seq_var.append(data)
            self.seq_var_map[slot] = data
            self.data_var.append(data)
        return self.data_var

    def get_dataloader(self, data_path):
        self.data_loader = fluid.io.DataLoader.from_generator(
            feed_list=self.data_var,
            capacity=self.capacity,
            use_double_buffer=True,
            iterable=self.iterable
        )
        # self.data_loader.set_batch_generator(reader, places=self.places)
        reader = fluid.io.batch(self.get_reader(data_path), batch_size=self.batch_size)
        reader = fluid.io.shuffle(reader, buf_size=self.batch_size*2)
        self.data_loader.set_sample_list_generator(reader, places=self.places)
        return self.data_loader

    def get_slot2index(self):
        return self.slot2index

    def get_slots(self):
        return self.total_slots

    def get_hash_size(self):
        return self.hash_size

    def get_data_var(self):
        return self.dense_var, self.sparse_var, self.seq_var

    def get_label(self):
        return self.label


if __name__ == "__main__":
    dense_slots = "I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,I13".split(',')
    sparse_slots = ["feat_idx"]
    places = fluid.CPUPlace()
    reader_cls = ReaderBase(sparse_slots=sparse_slots, dense_slots=dense_slots, places=places)
    reader = reader_cls.get_reader('../data_files/train_data')
    for i in range(1):
        reader()
