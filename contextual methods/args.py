
args = {
    'data_dir': 'data/',
    'model_type': 'bert',
    'model_name': 'bert-base-cased', #'./new_model',   # 'bert-base-cased'
    'task_name': 'binary',
    'output_dir': 'outputs/',
    'cache_dir': 'cache/',
    'do_train': True,
    'do_eval': True,
    'fp16': False,
    'fp16_opt_level': 'O1',
    'max_seq_length': 128,
    'output_mode': 'classification',
    'train_batch_size': 8,
    'eval_batch_size': 8,

    'gradient_accumulation_steps': 1,
    'num_train_epochs': 1,  # self.epoch, # 1
    'weight_decay': 0,
    'learning_rate': 4e-5,
    'adam_epsilon': 1e-8,
    'warmup_steps': 0,
    'max_grad_norm': 1.0,

    'logging_steps': 50,
    'evaluate_during_training': True,
    'save_steps': 2000,
    'eval_all_checkpoints': True,

    'overwrite_output_dir': True,
    'reprocess_input_data': True,
    'notes': 'Using Yelp Reviews dataset'
}