import pickle
from urllib.parse import urlparse

import pandas as pd
import torch
from flask import Flask, render_template, request

from model import load_bert, ClassifierModel
from predict import pred_df
from utils import get_ratings, check_domain_link

# Define Web-App with Flask API
app = Flask(__name__, template_folder="templates")

# Check device
device = torch.device("cuda" if torch.cuda.is_available else "cpu")

print("Loading Model...")
with open("../model/tokenizer.pkl", "rb") as f:
    dictionary = pickle.load(f)
    f.close()

# Load model
tokenizer, phoBert = load_bert()
model = torch.load("/home/nguyenduy/PycharmProjects/sentiment-ratings/app/model/model1.pt", map_location=device)

# Home API
@app.route("/")
def home():
    return render_template("index.html", title="Home")


# Process API
@app.route("/process", methods=["POST", "GET"])
def process():
    # Check error link
    if not request.form["link"]:
        return render_template("index.html", title="Home", error="Chưa nhập đường dẫn của sản phẩm")

    if not check_domain_link(request.form["link"]):
        return render_template("index.html", title="Home", error="Xin hãy nhập đường dẫn của sản phẩm trên Shopee")

    # Call API to get ratings from product link
    product_items, filtered_data, raw_data = get_ratings(request.form["link"], max_cmt=request.form["max_cmt"])

    # Classify data frame
    predicted_data = pred_df(filtered_data, model, tokenizer)
    raw_data["predicted"] = predicted_data["predicted"]

    # Extract each class
    bad_ratings = raw_data.where(raw_data["predicted"] == 0).dropna()
    med_ratings = raw_data.where(raw_data["predicted"] == 1).dropna()
    gud_ratings = raw_data.where(raw_data["predicted"] == 2).dropna()
    spm_ratings = raw_data.where(raw_data["predicted"] == 3).dropna()

    # Get product info
    product_name = product_items[0]["name"]
    product_image = "https://cf.shopee.vn/file/" + product_items[0]["image"]

    """
    Option:
        save_data = filtered_data[filtered_data["predicted"] != 2]
        file_path = "../data/data2.csv"
        df.to_csv(file_path, mode="a", index=False, header=False, encoding="utf-8")
    """

    # result = {
    #     "title": "Product",
    #     "total": len(filtered_data),
    #     "good": gud_ratings["comments"],
    #     "medium": med_ratings["comments"],
    #     "bad": bad_ratings["comments"],
    #     "spam": spm_ratings["comments"],
    #     "name": product_name,
    #     "image": product_image
    # }

    return render_template(
        "index.html",
        title="Product",
        total=len(filtered_data),
        good=gud_ratings["comments"],
        medium=med_ratings["comments"],
        bad=bad_ratings["comments"],
        spam=spm_ratings["comments"],
        name=product_name,
        image=product_image,
    )


@app.route("/product")
def about_product():
    return render_template("product_detail.html")

app.run(port=5000, host="0.0.0.0")
