str_config = {
  'loss': 'ccl',
  'lr': 0.0005,
  'ccl_neg_num': 300,
  'ccl_neg_margin': 0.4,
  'ccl_neg_weight': 50,
  'weight_decay': 0.000001,
  'batch_size': 1024,
  'n_epoch': 50,
  'latent_dim': 128,
  'affinity': 'dot',
  'aggregate': 'weighted-sum',
  'aggregate_w': 0.8,
  'dropout': 0.1,
  'n_interactive_items': 16,
  'popular_alpha': 0.2,
  'group_loss_gamma': 0.2,
  'item_dropout': 0.8,
}

str_config_yelp = {
  'loss': 'ccl',
  'lr': 1e-4,
  'ccl_neg_num': 300,
  'ccl_neg_margin': 0.4,
  'ccl_neg_weight': 50,
  'weight_decay': 1e-6,
  'batch_size': 512,
  'n_epoch': 100,
  'latent_dim': 256,
  'affinity': 'dot',
  'aggregate': 'mean',
  # 'attention_head': 8,
  'aggregate_w': 0.8,
  'dropout': 0.1,
  'n_interactive_items': 32,
  'popular_alpha': 0.1,
  'item_dropout': 0.9,
}

str_config_best_yelp = {
  'loss': 'ccl',
  'lr': 1e-4,
  'ccl_neg_num': 300,
  'ccl_neg_margin': 0.4,
  'ccl_neg_weight': 50,
  'weight_decay': 1e-6,
  'batch_size': 512,
  'n_epoch': 100,
  'latent_dim': 256,
  'affinity': 'cos',
  'aggregate': 'self-attention',
  'attention_head': 8,
  'aggregate_w': 0.8,
  'dropout': 0.1,
  'n_interactive_items': 32,
  'popular_alpha': 0.1,
  'item_dropout': 0.9,
}


str_config_gowalla = {
  'loss': 'ccl',
  'lr': 0.003,
  'ccl_neg_num': 128,
  'ccl_neg_margin': 0.1,
  'ccl_neg_weight': 1,
  'weight_decay': 1e-5,
  'batch_size': 512,
  'n_epoch': 10,
  'latent_dim': 64,
  'affinity': 'cos',
  'n_interactive_items': 8,
  'aggregate': 'mean',
  'attention_head': 2,
  'aggregate_w': 1
}

simplex_config = {
  'loss': 'ccl',
  'lr': 0.0001,
  'ccl_neg_num': 1000,
  'ccl_neg_margin': 0.9,
  'ccl_neg_weight': 150,
  'weight_decay': 1e-8,
  'batch_size': 512,
  'n_epoch': 10,
  'latent_dim': 64,
  'affinity': 'cos',
  'aggregate': 'mean',
  'aggregate_w': 0
}

mf_bpr_config = {
  'loss': 'bpr',
  'lr': 0.001,
  'ccl_neg_num': 1000,
  'ccl_neg_margin': 0.9,
  'ccl_neg_weight': 150,
  'weight_decay': 1e-4,
  'batch_size': 512,
  'n_epoch': 10,
  'latent_dim': 64,
  'affinity': 'cos',
  'aggregate': 'mean',
  'aggregate_w': 0
}


simplex_config_bak = {
  'loss': 'ccl',
  'lr': 1e-4,
  'ccl_neg_num': 1000,
  'ccl_neg_margin': 0.9,
  'ccl_neg_weight': 150,
  'weight_decay': 1e-8,
  'batch_size': 512,
  'n_epoch': 50,
  'latent_dim': 64,
  'affinity': 'cos',
  'aggregate': 'mean',
  'aggregate_w': 1,
  'dropout': 0.1
}

str_config_yelp = {
  'loss': 'ccl',
  'lr': 1e-4,
  'ccl_neg_num': 100,
  'ccl_neg_margin': 0.5,
  'ccl_neg_weight': 50,
  'weight_decay': 1e-6,
  'batch_size': 512,
  'n_epoch': 100,
  'latent_dim': 128,
  'affinity': 'cos',
  'aggregate': 'self-attention',
  'attention_head': 16,
  'aggregate_w': 0,
  'dropout': 0.1,
  'n_interactive_items': 32,
}

simplex_config_1 = {
  'loss': 'ccl',
  'lr': 1e-4,
  'ccl_neg_num': 100,
  'ccl_neg_margin': 0.5,
  'ccl_neg_weight': 50,
  'weight_decay': 1e-6,
  'batch_size': 512,
  'n_epoch': 100,
  'latent_dim': 128,
  'affinity': 'cos',
  'aggregate': 'mean',
  'attention_head': 8,
  'aggregate_w': 0,
  'dropout': 0.1,
  'n_interactive_items': 8,
}


sweep_configuration = {
  'method': 'grid',
  'name': 'MF-CCL-YELP',
  'metric': {'goal': 'maximize', 'name': 'Recall'}, 
  'parameters': {key: {'value': value} for key, value in str_config.items()}
}

sweep_configuration['parameters']['group_loss_gamma'] = {'values': [0, 0.1, 0.2, 0.3, 0.4, 0.5]}
sweep_configuration['parameters']['aggregate'] = {'values': ['weighted-sum']}
