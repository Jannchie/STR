#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from models import MF, STR, SimpleX
import torch
from config import str_config, simplex_config
from utils import RecHelper, load_bili_2
sns.set_theme(style="ticks", rc= { 'figure.figsize': (11.7,8.27),})
sns.set_context("paper", font_scale=2)
sns.set_palette("Set2")
from sklearn.decomposition import PCA

#%%
popular_alpha_df = pd.read_csv('results/popular_alpha.csv')
popular_alpha_df.to_latex('results/popular_alpha.tex', float_format='%.3f', bold_rows=True)
plt.ylim(0.57, 0.6)
sns.barplot(data=popular_alpha_df, x='popular_alpha', y='Recall', linewidth=1, edgecolor=".0", color='C0')
plt.ylabel('Recall@20')
plt.xlabel('popular alpha')
plt.savefig('results/popular_alpha.svg', format='svg', dpi=1200)
#%%
path_simplex  ='output/SimpleX-bilibili_result_20221228121203.csv'
path_str = 'output/STR-bilibili_result_20221231002129.csv'
path_mf = 'output/MF-bilibili_result_20221228234329.csv'
df_simplex = pd.read_csv(path_simplex, index_col=0)
df_str = pd.read_csv(path_str, index_col=0)
df_mf = pd.read_csv(path_mf, index_col=0)
dfs = []
for metric in ['Recall', 'Precision', 'NDCG', 'MAP', 'F1']:
  dfs.extend((
      [metric, df_str[metric].mean(), 'STR'],
      [metric, df_simplex[metric].mean(), 'SimpleX'],
      [metric, df_mf[metric].mean(), 'MF']
  ))
df = pd.DataFrame(dfs, columns=['Metric', 'Value', 'Model'])

sns.barplot(data=df,x="Metric", y="Value", hue="Model",linewidth=1, edgecolor=".0")
# save svg
plt.savefig('output/STR_VS_SimpleX_VS_MF.svg', format='svg', dpi=1200)

#%%
df_mean = [
  df_mf.mean(),
  df_simplex.mean(),
  df_str.mean(),
]
improve = (df_mean[2] - df_mean[1]) / df_mean[1]
df_mean.append(improve)
new_df = pd.DataFrame(df_mean, index=['STR', 'SimpleX', 'MF', 'Improve'])
new_df.to_latex('output/STR_VS_SimpleX_VS_MF.tex', float_format='%.3f', bold_rows=True)

#%%
df = pd.read_csv('results/author_tag_rec_feedbacks.csv', index_col=0)
sns.violinplot(data=df, x='method', y='relevance', hue="method", linewidth=1, edgecolor=".0", bw=.3, inner="quart", split=True)
plt.ylim(0, 5)

#%%
df = pd.read_csv('results/author_tag_rec_recognitions.csv', index_col=0)
# precent bar stack
df = df[df['method'] != 'test']
df = df[df['recognition'] != 0]
# created_at < 2023-01-01
df = df[df['created_at'] < '2022-12-30']
dfs = df.groupby(['m_id', 'recognition', 'method']).size().unstack()
dfs = dfs[dfs['str'].notna() & dfs['usage'].notna()]
dfs = df.groupby(['method', 'recognition']).size().unstack()
dfs = dfs.div(dfs.sum(axis=1), axis=0)
dfs.plot.bar(stacked=True, color=['C1', 'C0'])
dfs
#%%
dfs = []
for metric in ['Recall', 'Precision', 'NDCG', 'MAP', 'F1']:
  dfs.extend([metric, row, 'SimpleX'] for row in df_simplex[metric])
  dfs.extend([metric, row, 'STR'] for row in df_str[metric])
df = pd.DataFrame(dfs, columns=['Metric', 'Value', 'Type'])
# Load the example tips dataset
tips = sns.load_dataset("tips")

sns.barplot(data=df,x="Metric", y="Value", hue="Type",
               split=True, inner="quart", bw=.3, cut=0, linewidth=1)
plt.ylim(0, 1)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
# sns.despine(left=True) 
#%%
path = 'output/STR-gowalla_result_20230107015734.csv'
pd.read_csv(path,index_col=0)

#%%
pd.read_csv('results/yelp.csv').to_latex('results/yelp.tex', float_format='%.3f', bold_rows=True, index=False)

#%%
df = pd.read_csv('results/speed.csv')
df.to_latex('results/speed.tex', float_format='%.3f', bold_rows=True, index=False)
# %%
data = load_bili_2()
#%%
model_str = STR(data, config=str_config)
helper = RecHelper(model_str, data)
model_str.load_state_dict(torch.load('./output/STR-bilibili_model_best.pth'))
model_mf = MF(data, config={'latent_dim': 64, 'lr': 0.001, 'weight_decay': 0.0001, 'batch_size': 1024, 'epochs': 100, 'device': 'cuda', 'verbose': True, 'early_stop': True, 'patience': 10, 'save_path': './output/MF-bilibili_model.pth', 'load_path': None, 'seed': 2021})
model_mf.load_state_dict(torch.load('./output/MF-bilibili_model.pth'))
simplex_config['latent_dim'] = 128
model_simplex = SimpleX(data, config=simplex_config)
model_simplex.load_state_dict(torch.load('./output/SimpleX-bilibili_model.pth'))
model_simplex.load_state_dict(torch.load('./output/SimpleX-bilibili_model.pth'))
#%%
users = ['1850091', '585267', '423895', '40665101', '401742377']
user_idxs = torch.tensor([model_str.helper.uid2idx[i] for i in users])
item_idxs = torch.tensor(model_str.helper.train_set[model_str.helper.train_set['user_id'].isin(user_idxs)].item_id.unique(), device=model_str.device)
user_idxs = user_idxs.to(model_str.device)
#%%
def generate_pca(model):
  if model.__class__.__name__ in ['MF']:
    user_embs = model.W.data[user_idxs]
    item_embs = model.H.data[item_idxs]
  else:
    user_embs = model.item_embedding(user_idxs)
    item_embs = model.item_embedding(item_idxs)
  joint_embs = torch.cat([item_embs, user_embs], dim=0)
  model.pca = PCA(n_components=2).fit_transform(joint_embs.cpu().detach().numpy())
for model in [model_mf, model_str, model_simplex]:
  generate_pca(model)
# x, y = pca_user_point[:,0], pca_user_point[:,1]
# # Draw a combo histogram and scatterplot with density contours
# f, ax = plt.subplots(figsize=(12, 12))
# ax.set_xlim(-0.5, 0.5)
# ax.set_ylim(-0.5, 0.5)
# sns.scatterplot(x=x, y=y, s=5, color=".15")
# sns.histplot(x=x, y=y, bins=100, pthresh=.1, cmap="mako")
# sns.kdeplot(x=x, y=y, levels=5, color="w", linewidths=1)
sns.set_palette("Set2")

def get_item_points(idx, m):
  global items
  mask = model_str.user_top_index[idx] < model_mf.nitem
  masked_idx = model_str.user_top_index.cpu()[idx][mask]
  # map by item_idxs
  masked_idx = torch.tensor([item_idxs.tolist().index(i) for i in masked_idx], device=model_str.device)
  items = m.pca[masked_idx.cpu()]
  counts = model_str.user_top_count[idx].cpu()
  s = counts[mask]
  # items = (0.5-(-0.5))*(items-items.min())/(items.max()-items.min()) + (-0.5)
  x, y = items[:,0], items[:,1]
  return x, y, s * 40
def get_user_point(idx, m):
  idx = torch.tensor([user_idxs.tolist().index(idx)], device=model_str.device)
  return m.pca_u[idx, 0], m.pca_u[idx, 1]

# multi plot
fig = plt.figure(figsize=(17, 5))
for m in [model_mf, model_simplex,  model_str]:
  ax = fig.add_subplot(1, 3, 1 if m.__class__.__name__ == 'MF' else 2 if m.__class__.__name__ == 'SimpleX' else 3)
  ax.set_title(m.__class__.__name__)
  for i in users:
    idx = data.uid2idx[i]
    x, y, s = get_item_points(idx, m)
    sns.scatterplot(x=x, y=y, s=s, alpha=.8, linewidth=1.5, edgecolor='.2', palette='Set1')
# to svg
fig.savefig('results/pca.svg', format='svg', dpi=1200)
# %%

# pandas stack df
datasets = [helper.data.train_set, helper.data.test_set]
df = pd.concat(datasets, keys=['train', 'test']).reset_index(level=1, drop=True).reset_index()
df.groupby(['user_id', 'item_id']).count()
df.count()
# %%
