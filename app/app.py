import pandas as pd
from flask import Flask, render_template, request
from predict import pred_df
from utils import get_ratings
from model import load_bert, ClassifierModel
import torch
import pickle

# Define Web-App with Flask API
app = Flask(__name__, template_folder='templates')

# Check device
device = torch.device("cuda" if torch.cuda.is_available else "cpu")

with open('../model/tokenizer.pkl', 'rb') as f:
    dictionary = pickle.load(f)
    f.close()

# Load model
tokenizer, phoBert = load_bert()
model = torch.load("../model/model.pt", map_location=torch.device(device))


# Home API
@app.route('/')
def home():
    return render_template("index.html")


# Process API
@app.route('/process', methods=['POST', 'GET'])
def process():
    # Call API to get ratings from product link
    product_items, filtered_data, raw_data = get_ratings(request.form["link"])
    print(product_items)

    predicted_data = pred_df(filtered_data, model, tokenizer)
    filtered_data["predicted"] = predicted_data["predicted"]

    bad_ratings = filtered_data.where(filtered_data["predicted"] == 0).dropna()
    med_ratings = filtered_data.where(filtered_data["predicted"] == 1).dropna()
    gud_ratings = filtered_data.where(filtered_data["predicted"] == 2).dropna()
    spm_ratings = filtered_data.where(filtered_data["predicted"] == 3).dropna()

    product_name = product_items[0]["name"]
    product_image = "https://cf.shopee.vn/file/" + product_items[0]["image"]

    print(product_name, product_image)

    return render_template("index.html",
                           total=len(filtered_data),
                           good=gud_ratings["comments"],
                           medium=med_ratings["comments"],
                           bad=bad_ratings["comments"],
                           spam=spm_ratings["comments"],
                           name=product_name,
                           image=product_image
                           )


app.run(port=5000, host='0.0.0.0')

