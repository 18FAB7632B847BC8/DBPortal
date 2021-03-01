#!/usr/bin/env python

simple_model_config = {
    'seed': 1235,
    'token_num': 50265,
    'bert_layer_num': 4,
    'use_dropout': True,
    'use_dropout_bert': True,
    'lrate': 1e-5,
    'lrate_bert': 3e-7,
    'clip_c': 1.,
    'bert_last_num': 1,
    'save_path': '../data/save/simple_bert',
    'dropout_rate': 0.2,
    'dropout_rate_bert': 0.1,
    'warmup_steps': 10000,
    'decay_steps': 1e+7
}

simple_train_config = {
    "input_path": "../data/build/build_train.inputs",
    "output_path": "../data/build/build_train.outputs",
    "partition": "train",
    "batch_size": 12,
    "shuffle_mode": "simple"
}

simple_val_config = {
    "input_path": "../data/build/simple_dev/build_dev.inputs",
    "output_path": None,
    "partition": "dev",
    "batch_size": 1,
    "shuffle_mode": None
}

simple_val_loss_config = {
    "input_path": "../data/build/build_dev.inputs",
    "output_path": "../data/build/build_dev.outputs",
    "partition": "train",
    "batch_size": 4,
    "shuffle_mode": None
}

trainer_config = {
    'save_best_n': 10,
    'max_epochs': 300,
    'patience': 1000,
    'valid_start': 0,
    'metrics': ['loss', 'accuracy'],
    'early_metric': 'accuracy',
    'f_valid': 500,
    'f_verbose': 100,
    'dump_frequency': 2000
}
