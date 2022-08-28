
import math
from paddle import fluid
from models.base import Model

class DeepFM(Model):
    def __init__(self, num_field, embed_dim, reg=1e-4, layer_sizes=[400, 400, 400], act="relu", init_value=0.1, **kwargs):
        self.num_field = num_field
        self.embed_dim = embed_dim
        self.reg = reg
        self.layer_sizes = layer_sizes
        self.act = act
        self.init_value = init_value
        super(DeepFM, self).__init__(**kwargs)

    def net(self, input, is_infer=False):
        label = self.get_label()
        dense_var, sparse_var, _ = self.get_inputs()
        hash_size = self.get_hash_size()
        first_weights = fluid.embedding(input=sparse_var[0],
                                     is_sparse=True,
                                     dtype="float32",
                                     size=[hash_size+1, 1],
                                     padding_idx=0,
                                     param_attr=fluid.ParamAttr(initializer=fluid.initializer.TruncatedNormalInitializer(
                                         loc=0.0, scale=self.init_value),
                                         regularizer=fluid.regularizer.L1DecayRegularizer(self.reg)))
        first_weights = fluid.layers.reshape(first_weights, shape=[-1, self.num_field, 1])
        y_first_order = fluid.layers.reduce_sum(first_weights, 1)

        embeddings = fluid.embedding(input=sparse_var[0],
                                     is_sparse=True,
                                     dtype="float32",
                                     size=[hash_size+1, self.embed_dim],
                                     padding_idx=0,
                                     param_attr=fluid.ParamAttr(initializer=fluid.initializer.TruncatedNormalInitializer(
                                         loc=0.0, scale=self.init_value/math.sqrt(float(self.embed_dim)))))

        second_weights = fluid.layers.reshape(embeddings, shape=[-1, self.num_field, self.embed_dim])
        sum_emb = fluid.layers.reduce_sum(second_weights, 1)
        sum_emb_square = fluid.layers.square(sum_emb)
        emb_square = fluid.layers.square(second_weights)
        emb_square_sum = fluid.layers.reduce_sum(emb_square, 1)
        y_second_order = 0.5 * fluid.layers.reduce_sum((sum_emb_square - emb_square_sum), 1, keep_dim=True)

        dnn_sparse = fluid.layers.reshape(
            embeddings, [-1, self.num_field * self.embed_dim])
        dnn_dense = fluid.layers.concat(dense_var, 1)
        y_dnn = fluid.layers.concat([dnn_sparse, dnn_dense], 1)

        for i, s in enumerate(self.layer_sizes):
            y_dnn = fluid.layers.fc(
                name="fc_{}".format(i),
                input=y_dnn,
                size=s,
                act="relu",
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.TruncatedNormalInitializer(
                        loc=0.0, scale=self.init_value / math.sqrt(float(10)))),
                bias_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.TruncatedNormalInitializer(
                        loc=0.0, scale=self.init_value)))
        y_dnn = fluid.layers.fc(
            input=y_dnn,
            size=1,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormalInitializer(
                    loc=0.0, scale=self.init_value)),
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormalInitializer(
                    loc=0.0, scale=self.init_value)))

        output = y_first_order + y_second_order + y_dnn
        self.logits = fluid.layers.sigmoid(output)

        cost = fluid.layers.log_loss(input=self.logits, label=fluid.layers.cast(label, "float32"))
        avg_cost = fluid.layers.reduce_mean(cost)
        self.cost = avg_cost

        predict_val = fluid.layers.concat([1 - self.logits, self.logits], 1)
        label_int = fluid.layers.cast(label, 'int64')
        auc_val, batch_auc_var, auc_states = fluid.layers.auc(input=predict_val, label=label_int, slide_steps=0)

        self.metrics["AUC"] = auc_val
        self.metrics["BATCH_AUC"] = batch_auc_var
        self.metrics["loss"] = self.cost
        self.auc_states = auc_states

        return self.cost