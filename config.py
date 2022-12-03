str_config = {
  'loss': 'ccl',
  'lr': 0.003,
  'ccl_neg_num': 500,
  'ccl_neg_margin': 0,
  'ccl_neg_w': 1.1,
  'weight_decay': 1e-5,
  'batch_size': 1024,
  'n_epoch': 10,
  'latent_dim': 256,
  'affinity': 'dot',
  'n_interactive_items': 16,
  'aggregate': 'self-attention',
  'attention_head': 4,
  'aggregate_w': 0.8
}

str_config_yelp = {
  'loss': 'ccl',
  'lr': 0.0001,
  'ccl_neg_num': 64,
  'ccl_neg_margin': 0.2,
  'ccl_neg_w': 1,
  'weight_decay': 1e-5,
  'batch_size': 512,
  'n_epoch': 50,
  'latent_dim': 256,
  'affinity': 'cos',
  'n_interactive_items': 16,
  'aggregate_w': 0,
  'aggregate': 'self-attention',
  'attention_head': 1,
}

str_config_gowalla = {
  'loss': 'ccl',
  'lr': 0.003,
  'ccl_neg_num': 128,
  'ccl_neg_margin': 0.1,
  'ccl_neg_w': 1,
  'weight_decay': 1e-5,
  'batch_size': 512,
  'n_epoch': 10,
  'latent_dim': 64,
  'affinity': 'cos',
  'n_interactive_items': 16,
  'aggregate': 'mean',
  'attention_head': 2,
  'aggregate_w': 1
}

simplex_config_gowalla = {
  'loss': 'ccl',
  'lr': 0.003,
  'ccl_neg_num': 1000,
  'ccl_neg_margin': 0.5,
  'weight_decay': 1e-8,
  'batch_size': 512,
  'n_epoch': 10,
  'latent_dim': 64,
  'affinity': 'dot',
  'aggregate': 'mean',
  'aggregate_w': 0.5
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


simplex_config = {
  'loss': 'ccl',
  'lr': 1e-4,
  'ccl_neg_num': 1000,
  'ccl_neg_margin': 0.9,
  'ccl_neg_weight': 150,
  'weight_decay': 1e-8,
  'batch_size': 512,
  'n_epoch': 100,
  'latent_dim': 128,
  'affinity': 'cos',
  'aggregate': 'mean',
  'aggregate_w': 1,
  'dropout': 0.2,
  'n_interactive_items': None,
}


sweep_configuration = {
    'method': 'bayes',
    'name': 'STR-YELP-ATTENTION-1024-10-3',
    'metric': {'goal': 'maximize', 'name': 'Recall'},
    'parameters': 
    {
        'loss': {'value': 'ccl'},
        'lr': {'value': 0.001 },
        'ccl_neg_num': {'values': [64, 128, 256, 512]},
        'ccl_neg_margin': { 'values': [x/10 for x in range(10)] },
        'ccl_neg_w': {'values': [x/10 for x in range(5, 15)]},
        'weight_decay': {'value': 1e-5 },
        'batch_size':  {'value': 512 },
        'n_epoch':  {'value': 3},
        'latent_dim': {'values': [32, 64, 128]},
        'affinity': {'value': 'dot'},
        'n_interactive_items': { 'values': [1, 4, 8, 16, 32] },
        'aggregate':  {'value':'self-attention'},
        'attention_head': {'value': 1},
        'aggregate_w': {'values': [0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1] },
     }
}