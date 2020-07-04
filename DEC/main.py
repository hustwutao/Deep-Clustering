import tensorflow as tf
from model import DEC
from trainer import Trainer
from utils import get_config_from_json

def main():

    config = get_config_from_json('config.json')
    # create an instance of the model
    model = DEC(config)
    # create trainer instance
    trainer = Trainer(config, model)
    # train the model
    trainer.train()

if __name__ == '__main__':
    main()
