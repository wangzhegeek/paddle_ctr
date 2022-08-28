
import os
import random
import paddle.fluid as fluid
from config import parse_args
from runner import Runner
from models.deepfm import DeepFM
from instance.base import ReaderBase


cont_min_ = [0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
cont_max_ = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
cont_diff_ = [20, 603, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
continuous_range_ = range(1, 14)
categorical_range_ = range(14, 40)
dense_slots = "I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,I13".split(',')
sparse_slots = "C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15,C16,C17,C18,C19,C20,C21,C22,C23,C24,C25,C26".split(',')


def make_sample(data_path, train_path):
    f_train = open(train_path, 'w')
    cnt = 0
    for line in open(data_path):
        line_info = line.strip('\n').split('\t')
        print(line_info)
        print(len(line_info))
        cnt += 1
        if cnt == 10:
            break
        ins = []
        for i, item in enumerate(line_info):
            if i == 0:
                out_item = 'label:' + item
            elif i in continuous_range_:
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
    # dense_slots, sparse_slots = get_slots(args.raw_data)
    places = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    data_class = ReaderBase(places=places, sparse_slots=['feat_idx'], dense_slots=dense_slots, batch_szie=args.batch_size, hash_size=args.hash_size)
    model = DeepFM(save_path=args.model_dir, data_class=data_class, lr=args.lr, num_field=len(sparse_slots), embed_dim=args.embed_dim, layer_sizes=args.layer_sizes,
                    act=args.act)
    runner = Runner(model=model, dataset=data_class, places=places, epoch=args.epochs, train_path=args.train_data, test_path=args.test_data, model_dir=args.model_dir, optimizer=args.optimizer)
    runner.run_train()
    print("train done")
    # runner.run_test(test_epoch=30)
