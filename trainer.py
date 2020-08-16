import os
import yaml
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import nn
from paddle.fluid.dygraph import to_variable



class DefaultTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_dict = {}
        self.exclude_save = []
        self.optimizers = []
        self.iterations = 0

        self.build_models(cfg)
        self.setup(cfg)

    def build_models(self, cfg):
        raise NotImplementedError

    def step(self, inputs):
        raise NotImplementedError

    def setup(self, cfg):
        self.output_dir = cfg.output_dir
        self.ckpt_dir = os.path.join(self.output_dir, 'checkpoints')
        self.output_image_dir = os.path.join(self.output_dir, 'images')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.output_image_dir, exist_ok=True)

        with open(os.path.join(self.output_dir, 'config.yaml'), 'w') as f:
            yaml.dump(cfg.copy(), f)

        self.load_weights(cfg)

    def load_weights(self, cfg):
        for name, model in self.model_dict.items():
            key = 'weights_' + name
            if cfg.get(key):
                sd, _ = fluid.load_dygraph(cfg.get(key))
                model.set_dict(sd)
                print('load', key, 'from', cfg.get(key))

    def save_checkpoints(self, niter):
        for name, model in self.model_dict.items():
            fluid.save_dygraph(model.state_dict(), os.path.join(self.ckpt_dir, '{}_{}'.format(name, niter)))

    def save_last_checkpoints(self, epoch):
        with open(os.path.join(self.ckpt_dir, 'last_checkpoint'), 'w') as f:
            f.write('%d' % epoch)
        self.save_checkpoints('last')

    def resume(self):
        for name, model in self.model_dict.items():
            weight_path = os.path.join(self.ckpt_dir, name+'_last')
            sd, _ = fluid.load_dygraph(weight_path)
            model.set_dict(sd)
            print('resume weight', name, 'from', weight_path)

        with open(os.path.join(self.ckpt_dir, 'last_checkpoint'), 'r') as f:
            begin = int(f.readline()) + 1
        for optimizer in self.optimizers:
            optimizer._learning_rate.set_begin(begin)
        return  begin
