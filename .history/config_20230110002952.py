str_config = {
  'lr': 0.0005,
  'loss_neg_n': 300,
  'loss_neg_w': 50,
  'loss_neg_m': 0.4,
  'weight_decay': 1e-6,
  'batch_size': 1024,
  'n_epoch': 10,
  'latent_dim': 128,
  'affinity': 'dot',
  'aggregate': 'weighted-sum',
  'w_cf': .8,
  'w_h': .2,
  'w_g': .2,
  'dropout': .2,
  'n_interactive_items': 16,
  'popular_alpha': .2,
  'item_dropout': .8,
}

str_config_amazon = {
  'lr': 1e-4,
  'loss_neg_n': 300,
  'loss_neg_a': 1,
  'loss_neg_w': 50,
  'loss_neg_m': .4,
  'weight_decay': 1e-8,
  'batch_size': 512,
  'affinity': 'cos',
  'n_epoch': 100,
  'latent_dim': 256,
  'aggregate': 'weighted-sum',
  'w_cf': 0.9,
  'w_ii': 0.1,
  # 'w_uu': 0.0,
  'aggregate_a': 0,
  'n_interactive_items': 4,
  'popular_alpha': 0,
  'w_g': 0,
  'dropout': .1,
  'item_dropout': .1,
}

str_config_yelp = {
  'lr': 1e-4,
  'loss_neg_n': 300,
  'loss_neg_a': 1,
  'loss_neg_w': 50,
  'loss_neg_m': .4,
  'weight_decay': 0,
  'batch_size': 512,
  'affinity': 'dot',
  'n_epoch': 100,
  'latent_dim': 64,
  'aggregate': 'mean',
  'w_cf': 1.0,
  'w_ii': 0.0,
  'w_uu': 0.0,
  # 'w_uu': 0.0,
  'aggregate_a': 0,
  'n_interactive_items': 20,
  'popular_alpha': 0,
  'w_g': 0,
  'dropout': 0,
  'item_dropout': .0,
}

str_config_gowalla = {
  'lr': 1e-4,
  'loss_neg_n': 800,
  'loss_neg_a': 1,
  'loss_neg_w': 200,
  'loss_neg_m': .9,
  'weight_decay': 1e-8,
  'batch_size': 512,
  'affinity': 'cos',
  'n_epoch': 100,
  'latent_dim': 256,
  'aggregate': 'weighted-sum',
  'w_cf': 0.8,
  'w_ii': 0.2,
  # 'w_uu': 0.0,
  'aggregate_a': 0.05,
  'n_interactive_items': 4,
  'popular_alpha': 0,
  'w_g': 0,
  'dropout': .1,
  'item_dropout': .1,
}

simplex_config = {
  'loss': 'ccl',
  'lr': 0.0003,
  'loss_neg_n': 1000,
  'loss_neg_k': 0.9,
  'loss_neg_w': 150,
  'weight_decay': 1e-8,
  'batch_size': 1024,
  'n_epoch': 20,
  'latent_dim': 128,
  'affinity': 'cos',
  'n_interactive_items': 3,
  'aggregate': 'mean',
  'w_ii': 1,
  'dropout': 0.1,
}

mf_bpr_config = {
  'batch_size': 512,
  'n_epoch': 50,
  'latent_dim': 64,
  'lr': 0.003,
  'weight_decay': 1e-8,
}


simplex_config_bak = {
  'loss': 'ccl',
  'lr': 1e-4,
  'loss_neg_n': 1000,
  'loss_neg_k': 0.9,
  'loss_neg_w': 150,
  'weight_decay': 1e-6,
  'batch_size': 512,
  'n_epoch': 50,
  'latent_dim': 256,
  'affinity': 'cos',
  'aggregate': 'mean',
  'w_ii': 1,
  'dropout': 0.1
}

simplex_config_1 = {
  'loss': 'ccl',
  'lr': 1e-4,
  'loss_neg_n': 100,
  'loss_neg_k': 0.5,
  'loss_neg_w': 50,
  'weight_decay': 1e-6,
  'batch_size': 512,
  'n_epoch': 100,
  'latent_dim': 128,
  'affinity': 'cos',
  'aggregate': 'mean',
  'attention_head': 8,
  'w_ii': 0,
  'dropout': 0.1,
  'n_interactive_items': 8,
}

