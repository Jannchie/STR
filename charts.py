#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from models import STR
import torch
from config import str_config
from utils import RecHelper, load_bili_2
#%%
path_1  ='output/STR-bilibili_result_20221222235737.csv'
path_2 = 'output/STR-bilibili_result_20221223031958.csv'
df_1 = pd.read_csv(path_1, index_col=0)
df_2 = pd.read_csv(path_2, index_col=0)
dfs = []
for metric in ['Recall', 'Precision', 'NDCG', 'MAP']:
  dfs.extend([metric, row, '1'] for row in df_1[metric])
  dfs.extend([metric, row, '2'] for row in df_2[metric])
df = pd.DataFrame(dfs, columns=['Metric', 'Value', 'Type'])
df
#%%
sns.set_theme(style="whitegrid")
sns.set_palette("Set2")
# Load the example tips dataset
tips = sns.load_dataset("tips")

sns.violinplot(data=df,x="Metric", y="Value", hue="Type",
               split=True, inner="quart", bw=.2, cut=0)
sns.despine(left=True)
# sns.boxenplot(data=df,x="Metric", y="Value", hue="Type")
# sns.despine(left=True) 
# %%
data = load_bili_2()
model = STR(data, config=str_config)
helper = RecHelper(model, data)
model.load_state_dict(torch.load('./output/STR-bilibili_model.pth'))
#%%
from sklearn.decomposition import PCA
user_embs = model.item_embedding(torch.arange(model.nuser, device='cuda'))
pca_point = PCA(n_components=2).fit_transform(user_embs.cpu().detach().numpy())
sns.scatterplot(x=pca_point[:, 0], y=pca_point[:, 1])
