#!/usr/bin/env python

retrieval_model_config = {
    'seed': 1235,
    'weight_init_scale': 'xavier',
    'token_num': 50265,
    'margin': 0.3,
    'dropout_rate': 0.2,
    'dropout_rate_bert': 0.2,
    'use_dropout': True,
    'use_dropout_bert': True,
    'warmup_steps': 4000,
    'decay_steps': 300000,
    'lrate': 1e-4,
    'lrate_bert': 1e-6,
    'clip_c': 5.,
    'save_path': '../data/save/retrieval'
}

retrieval_simple_train_config = {
    'question_path': '../data/retrieval/wikisql/build/build_train.predict_value',
    'header_path': '../data/retrieval/wikisql/build/build_train.ground_truth_value',
    'batch_size': 24,
    'shuffle_mode': 'simple',
    'num_workers': 20,
}

retrieval_simple_valid_config = {
    'question_path': '../data/retrieval/wikisql/build/sub_build_test.predict_value',
    'header_path': '../data/retrieval/wikisql/build/sub_build_test.ground_truth_value',
    'batch_size': 24,
    'shuffle_mode': None,
    'num_workers': 20,
}


trainer_config = {
    'save_best_n': 10,
    'max_epochs': 200,
    'patience': 1000,
    'valid_start': 1,
    'metrics': [],
    'early_metric': 'loss',
    'f_valid': 1000,
    'f_verbose': 100,
    'dump_frequency': 2000
}
