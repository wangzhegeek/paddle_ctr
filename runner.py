
import os
import sys
import time
import logging
import numpy as np
import paddle.fluid as fluid

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('fluid')
logger.setLevel(logging.INFO)


class Runner(object):
    def __init__(self, model, dataset, places, epoch, train_path, test_path, model_dir, optimizer):
        self.model = model
        self.dataset = dataset
        self.places = places
        self.epoch = epoch
        self.train_path = train_path
        self.test_path = test_path
        self.model_dir = model_dir
        self.optimizer = optimizer

    def set_zero(self, var_name,
                 scope=fluid.global_scope(),
                 param_type="int64"):
        """
        Set tensor of a Variable to zero.
        Args:
            var_name(str): name of Variable
            scope(Scope): Scope object, default is fluid.global_scope()
            place(Place): Place object, default is fluid.CPUPlace()
            param_type(str): param data type, default is int64
        """
        param = scope.var(var_name).get_tensor()
        param_array = np.zeros(param._get_dims()).astype(param_type)
        param.set(param_array, self.places)

    def run_non_iterable(self, program, exe, metrics, data_loader, is_infer):
        data_loader.start()
        try:
            while True:
                metrics_vals = exe.run(program=program, fetch_list=list(metrics.values()))
                result_str = "\t".join([key + ": " + str(metrics_vals[i]) for i, key in enumerate(metrics.keys())])
                infer = "infer" if is_infer == True else "train"
                logger.info('{}: metrics is {}'.format(infer, result_str))
        except fluid.core.EOFException:
            print('End of epoch')
            data_loader.reset()

    def run_train(self):
        main_program = fluid.default_main_program()
        train_input = self.dataset.get_input_var()
        train_loader = self.dataset.get_dataloader(self.train_path)
        train_loss = self.model.net(train_input)
        optimizer = self.model.build_optimizer(self.optimizer)
        optimizer.minimize(train_loss)
        train_metrics = self.model.metrics

        exe = fluid.Executor(self.places)
        exe.run(fluid.default_startup_program())

        for epoch in range(self.epoch):
            start = time.time()
            self.run_non_iterable(main_program, exe, train_metrics, train_loader, is_infer=False)
            model_dir = os.path.join(self.model_dir,
                                     'epoch_' + str(epoch + 1))
            logger.info('epoch%d is finished and takes %f s\n' % ((epoch + 1), time.time() - start))
            fluid.io.save(main_program, model_dir)

    def run_test(self, test_epoch):
        inference_scope = fluid.Scope()
        startup_program = fluid.framework.Program()
        test_program = fluid.framework.Program()
        cur_model_path = os.path.join(self.model_dir,
                                      'epoch_' + str(test_epoch))
        with fluid.scope_guard(inference_scope):
            with fluid.framework.program_guard(test_program, startup_program):
                test_input = self.dataset.get_input_var()
                test_loader = self.dataset.get_dataloader(self.test_path)
                test_loss = self.model.net(test_input)
                test_metrics = self.model.metrics
                auc_states = self.model.auc_states

                exe = fluid.Executor(self.places)
                main_program = fluid.default_main_program()
                fluid.load(main_program, cur_model_path, exe)
                for var in auc_states:  # reset auc states
                    self.set_zero(var.name, scope=inference_scope)

                start = time.time()
                self.run_non_iterable(main_program, exe, test_metrics, test_loader, is_infer=True)
                logger.info('test takes %f s\n' % (time.time() - start))