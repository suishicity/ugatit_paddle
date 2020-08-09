import math
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.learning_rate_scheduler import LearningRateDecay


class LinearScheduler(LearningRateDecay):
    def __init__(self, learning_rate, epochs, decay_step=0, eta_min=1.0e-10, begin=0, step=1, dtype='float32'):
        super().__init__(begin, step, dtype)
        self.learning_rate = learning_rate
        self.decay_step = decay_step
        self.epochs = epochs
        self.eta_min = eta_min

    def set_begin(self, epoch):
        self.step_num = epoch
        self.begin = epoch

    def step(self):
        if self.step_num < self.decay_step:
            return self.create_lr_var(self.learning_rate)
        
        lr = self.learning_rate - (self.learning_rate - self.eta_min) * \
            (self.step_num - self.decay_step) / (self.epochs - self.decay_step)

        return self.create_lr_var(lr)