#%%
from thefuzz import fuzz
from thefuzz import process
# from utils import load_bili_2
import pandas as pd
import csv
from tqdm import tqdm

#%%
data = csv.reader(open('data/bilibili/video_data.csv', 'r', encoding='utf-8-sig'))
all_used_tag = [x[2:] for x in data]
# to one dim list
all_tags = []
for x in all_used_tag:
  all_tags.extend(x)
all_tags
#%%
# multiprocess
import multiprocessing as mp

df = pd.read_csv('data/bilibili/tag_info.csv')
t_list = [str(x) for x in df.tag_name.to_list()]
name_list = df.tag_name.to_list()
id_list = df.tag_id.to_list()
name_list.reverse()
id_list.reverse()
tag_name_2_tag_id = {str(x):y for x,y in zip(name_list, id_list)}
#%%
dict_list = t_list
#%%
id_mapper = {}
for tag in tqdm(dict_list):
  tag_id = tag_name_2_tag_id[tag]
  if tag_id in id_mapper:
    continue
  res = process.extract(tag, dict_list, limit=20, scorer=fuzz.ratio)
  same = [x[0] for x in res if x[1] >= 90]
  tag_id_list = [tag_name_2_tag_id[x] for x in same]
  for i in tag_id_list:
    if i not in id_mapper:
      id_mapper[i] = tag_id

#%%
import pickle
#%%


#%%
def tag_to_real_name(tag):
  true_id = id_mapper.get(df[df.tag_name==tag].tag_id.to_list()[0], df[df.tag_name==tag].tag_id.to_list()[0])
  return df[df.tag_id == true_id]

tag_to_real_name('#打卡挑战')