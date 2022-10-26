import re
import requests
import json
import pickle
import yaml
import pandas as pd

# Header will be deleted
HEADER = {
    "color": "màu sắc",
    "material": "chất liệu",
    "ship": "giao",
    "describe": "đúng với mô tả",
    "coin": "nhận xu"
}

# Number of rating will be got
MAX_RATINGS = 200

# Shopee API - Get Ratings
API_URL = "https://shopee.vn/api/v2/item/get_ratings?filter=0&flag=1&itemid=" \
          "{item_id}&limit=20&offset={offset}&shopid={shop_id}&type=0"

# Init some path will be used
tokenize_path = "model/tokenizer.pkl"

# Load dictionary file
with open(tokenize_path, 'rb') as f:
    dictionary = pickle.load(f)
    f.close()

with open("utils.yaml", 'r') as yaml_file:
    data = yaml.load(yaml_file, Loader=yaml.FullLoader)
    yaml_file.close()
header = data['HEADER']
print(header)