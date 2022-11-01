import pandas as pd
import torch
from model import load_bert, ClassifierModel
from utils import get_ratings
from predict import pred_df

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer, phoBert = load_bert()
model = ClassifierModel(phoBert)
model = torch.load("model/model.pt", map_location=torch.device(device))

url = "https://shopee.vn/Gi%C3%A0y-n%E1%BB%AF-gi%C3%A0y-da-n%E1%BB%AF-c%E1%BB%95-th%E1%BA%A5p-%C4%91%E1%BA%BF-%C4%91%E1%BB%99n-gi%E1%BA%A7y-th%E1%BB%9Di-trang-ki%E1%BB%83u-d%C3%A1ng-basic-ch%E1%BA%A5t-li%E1%BB%87u-da-PU-%C4%91%E1%BA%BF-cao-su-%C4%91%E1%BB%99n-4cm-L%C3%B9-store-GD02-i.404405029.20244927353?sp_atk=cb3eea42-18f8-4fac-b484-1263b4c9a7a0&xptdk=cb3eea42-18f8-4fac-b484-1263b4c9a7a0"

_, data, _ = get_ratings(url)
predicted_data = pred_df(data, model, tokenizer)

df = pd.DataFrame(predicted_data)

file_path = "data/data1.csv"
df.to_csv(file_path, mode='a', index=False, header=False, encoding="utf-8")

print(predicted_data)