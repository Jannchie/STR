from flask import Flask, jsonify
from models import STR, SimpleX
from utils import RecHelper, load_bili_2
from config import str_config
import torch
from thefuzz import process 
from thefuzz import fuzz
app = Flask(__name__)
helper: RecHelper = None

@app.route('/tag/<txt>')
def tag(txt):
  global helper
  return jsonify([{'score': x[1], 'value': x[0], 'id': int(helper.tag2id.get(x[0],0))} for x in process.extract(txt, helper.tags, limit=10, processor=lambda x: x, scorer=fuzz.ratio)])

@app.route('/<uid>')
def rec(uid: str):
  global helper
  res = helper.get_rec_tags(uid)
  return jsonify(res)

def init():
  global helper
  data = load_bili_2()
  model = STR(data, config=str_config).eval()
  helper = RecHelper(model, data)
  tag_dict = helper.tag_info
  helper.tags = list(helper.tag_info.values())[:10000]
  helper.tag2id = {helper.tag_info[k]:k for k in helper.tag_info}
  def search(txt):
    return [d for d in tag_dict if txt in d]
  model.load_state_dict(torch.load('./output/STR-bilibili_model_best.pth'))
  
if __name__ == '__main__':
  init()
  app.run(host='::', port=5001, debug=True)
else:
  init()
# waitress-serve --host 127.0.0.1 --port 5000 server:app