
import os
import random
import paddle.fluid as fluid
from config import parse_args
from runner import Runner
from models.deepfm import DeepFM
from instance.base import ReaderBase


continuous_range_ = range(0, 8)
categorical_range_ = range(8, 58+8)
cols = "education,marital_status,relationship,workclass,occupation,age_buckets,education_occupation,age_buckets_education_occupation,education_0,education_1,education_2,education_3,education_4,education_5,education_6,education_7,education_8,education_9,education_10,education_11,education_12,education_13,education_14,education_15,marital_status_0,marital_status_1,marital_status_2,marital_status_3,marital_status_4,marital_status_5,marital_status_6,relationship_0,relationship_1,relationship_2,relationship_3,relationship_4,relationship_5,workclass_0,workclass_1,workclass_2,workclass_3,workclass_4,workclass_5,workclass_6,workclass_7,workclass_8,occupation_0,occupation_1,occupation_2,occupation_3,occupation_4,occupation_5,occupation_6,occupation_7,occupation_8,occupation_9,occupation_10,occupation_11,occupation_12,occupation_13,occupation_14,age,education_num,capital_gain,capital_loss,hours_per_week,label"
dense_slots = cols.split(',')[:8]
sparse_slots = cols.split(',')[8:58+8]


def make_sample(data_path, train_path):
    f_train = open(train_path, 'w')
    for line in open(data_path):
        line_info = line.strip('\n').split(',')
        if line_info[0] == "education":
            continue
        ins = []
        out_item = 'label:' + line_info[-1]
        ins.append(out_item)
        for i, item in enumerate(line_info):
            if i in continuous_range_:
                off = continuous_range_[0]
                out_item = dense_slots[i-off] + ':' + item
            elif i in categorical_range_:
                out_item = 'feat_idx:' + item
            else:
                continue
            ins.append(out_item)
        ins_str = '\t'.join(ins)
        # rnd = random.uniform(0, 10)
        # if rnd <= test_size * 10:
        #     f_test.write(ins_str + '\n')
        # else:
        f_train.write(ins_str + '\n')
    return


if __name__ == "__main__":
    args = parse_args()
    make_sample(args.raw_train_data, args.train_data)
    make_sample(args.raw_test_data, args.test_data)
    places = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    data_class = ReaderBase(places=places, sparse_slots=['feat_idx'], dense_slots=dense_slots, batch_szie=args.batch_size, hash_size=args.hash_size)
    model = DeepFM(save_path=args.model_dir, data_class=data_class, lr=args.lr, num_field=len(sparse_slots), embed_dim=args.embed_dim, layer_sizes=args.layer_sizes,
                    act=args.act)
    runner = Runner(model=model, dataset=data_class, places=places, epoch=args.epochs, train_path=args.train_data, test_path=args.test_data, model_dir=args.model_dir, optimizer=args.optimizer)
    runner.run_train()
    print("train done")
    # runner.run_test(test_epoch=30)
