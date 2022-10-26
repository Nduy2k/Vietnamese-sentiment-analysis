import re
import requests
import pandas as pd
import pickle
from preprocessing import standardize_data, rm_header

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


def check_shopee_id(url):
    product_id = re.search(r"i\.(\d+)\.(\d+)", url)
    if product_id:
        return product_id
    else:
        return re.search(r"product/(\d+)/(\d+)", url)


def get_ratings(product_url, dictionary=None, header=None, max_cmt=200):
    """
    :param product_url: link of shopee product
    :param header: header will be removed
    :param max_cmt: number of max comments
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
    raw_ratings = []
    filtered_ratings = []
    product_items = {}

    while True:
        i = 1
        # Send requests
        response = requests.get(
            API_URL.format(shop_id=shop_id, item_id=item_id, offset=offset)
        ).json()

        ratings_data = response["data"]["ratings"]

        for i, rating in enumerate(ratings_data, 1):
            comments = standardize_data(rating['comment'])
            if comments:
                comments = rm_header(str(comments), header, dictionary)
                if comments:
                    filtered_ratings.append(comments)
                    raw_ratings.append(rating['comment'])
        if i % 20:
            break
        offset += 20

        product_data = ratings_data[0]
        product_items = product_data["product_items"]

        # if len(processed_data) > max_cmt:
        #     break

    filtered_ratings = pd.DataFrame(filtered_ratings, columns=['comments'])
    raw_ratings = pd.DataFrame(raw_ratings, columns=['comments'])

    return product_items, filtered_ratings, raw_ratings


if __name__ == "__main__":
    with open('model/tokenizer.pkl', 'rb') as f:
        DICTIONARY = pickle.load(f)
    f.close()
    url = "https://shopee.vn/D%C3%A2y-%C4%90eo-C%E1%BB%95-G%E1%BA%AFn-%C4%90i%E1%BB%87n-Tho%E1%BA%A1i-In-H%C3%ACnh-Ng%C3%B4i-Sao-B%C3%B3ng-%C4%90%C3%A1-B%C3%B3ng-%C4%90%C3%A1-i.263496460.16071769873?sp_atk=67b0a2fd-72c9-42af-801b-757aa289fc95&xptdk=67b0a2fd-72c9-42af-801b-757aa289fc95"
    print(get_ratings(url, dictionary=DICTIONARY))
