import os
import time
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from utils import data_loader, make_dirs

class Trainer(object):
    def __init__(self, config, model):
        self.config = config
        self.model = model

        # checkpoint
        #self.checkpoint_dir = make_dirs(os.path.join(self.config.result_path, self.config.checkpoint_path))
        #self.ckpt = tf.train.Checkpoint(enc=self.model.enc, dec=self.model.dec, optim=self.model.optim, epoch=self.model.global_epoch)
        #self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, directory=self.checkpoint_dir, checkpoint_name='ckpt', max_to_keep=2)

        # tensorboard
        self.tensorboard_dir = make_dirs(os.path.join(self.config.result_path, self.config.tensorboard_path))
        self.summary_writer = tf.summary.create_file_writer(self.tensorboard_dir)

    def train(self):
        self.model.pretrain()
        self.model.initialize()
        self.model.finetune()

   # save function that saves the checkpoint
    def save_model(self, epoch):
        print("Saving model...")
        self.ckpt_manager.save(checkpoint_number=epoch)
        print("Model saved")

    # load latest checkpoint
    def load_model(self):
        if self.ckpt_manager.latest_checkpoint:
            print("Restored from {}".format(self.ckpt_manager.latest_checkpoint))
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        else:
            print("Initializing from scratch.")
