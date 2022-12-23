from flask import Flask, jsonify
from models import STR, SimpleX
from utils import RecHelper, load_bili_2
from config import str_config
import torch

app = Flask(__name__)
helper = None


@app.route('/<uid>')
def hello(uid: str):
  res = helper.get_rec_tags(uid)
  return jsonify(res)


if __name__ == '__main__':
  data = load_bili_2()
  model = STR(data, config=str_config)
  helper = RecHelper(model, data)
  model.load_state_dict(torch.load('./output/STR-bilibili_model.pth'))
  app.run(host='::', port=5000, debug=True)
