import re

from unshortenit import UnshortenIt
import validators
import requests
import pandas as pd
import pickle
from preprocessing import standardize_data, rm_header
from urllib.parse import urlparse

HEADER = {
    "color": "màu sắc",
    "material": "chất liệu",
    "ship": "giao",
    "describe": "đúng với mô tả",
    "coin": "nhận xu"
}

with open('../model/tokenizer.pkl', 'rb') as f:
    DICTIONARY = pickle.load(f)
f.close()

API_URL = "https://shopee.vn/api/v2/item/get_ratings?filter=0&flag=1&itemid={item_id}&limit=20&offset={offset}&shopid={shop_id}&type=0"

def check_domain_link(link: str) -> bool:
    if not validators.url(link):
        return False

    unshortener = UnshortenIt()
    link = unshortener.unshorten(link)
    domain = urlparse(link).netloc

    if "shopee" in domain.split("."):
        return True
    else:
        return False


def check_shopee_id(link: str):
    unshortener = UnshortenIt()
    link = unshortener.unshorten(link)

    product_id = re.search(r"i\.(\d+)\.(\d+)", link)
    if product_id:
        return product_id
    else:
        return re.search(r"product/(\d+)/(\d+)", link)


def get_ratings(product_url: str, max_cmt: str, dictionary=None, header=None):
    """
    :param max_cmt: number of max comment
    :param product_url: link of shopee product
    :param header: header will be removed
    :param dictionary: dictionary
    :return: data frame comments
    """

    if header is None:
        header = HEADER

    if dictionary is None:
        dictionary = DICTIONARY

    # Get shop id and item id in product link
    product_id = check_shopee_id(product_url)
    shop_id, item_id = product_id[1], product_id[2]

    offset = 0
    count = 0
    raw_ratings = []
    filtered_ratings = []
    product_items = {}

    while True:
        i = 1
        # Send requests
        response = requests.get(
            API_URL.format(shop_id=shop_id, item_id=item_id, offset=offset)
        ).json()

        if response["data"]["ratings"] is None:
            break

        rcount_with_context = response["data"]["item_rating_summary"]["rcount_with_context"]

        for i, rating in enumerate(response["data"]["ratings"], 1):
            if rating["comment"]:
                count += 1

            comments = standardize_data(rating['comment'])
            if comments:
                comments = rm_header(str(comments), header, dictionary)
                if comments:
                    filtered_ratings.append(comments)
                    raw_ratings.append(rating['comment'])

            if count >= rcount_with_context:
                break

        if i % 20:
            break
        offset += 20

        if max_cmt != "All":
            if len(filtered_ratings) > int(max_cmt):
                break

        ratings_data = response["data"]["ratings"]
        product_data = ratings_data[0]
        product_items = product_data["product_items"]

    filtered_ratings = pd.DataFrame(filtered_ratings, columns=['comments'])
    raw_ratings = pd.DataFrame(raw_ratings, columns=['comments'])

    return product_items, filtered_ratings, raw_ratings


if __name__ == "__main__":
    print(check_shopee_id("https://shope.ee/7ezJaRHqka"))
