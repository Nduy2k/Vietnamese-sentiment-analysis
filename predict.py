import torch
import requests


from tqdm.notebook import tqdm
tqdm.pandas()

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

DEVICE = torch.device("cuda" if torch.cuda.is_available else "cpu")
print(DEVICE)


MAX_LEN = 50
CLASS_NAMES = ['Bad', 'Medium', 'Good', 'Spam']


def pred_sentence(sentence, model, tokenizer, device=None):
    """
    :param tokenizer:
    :param sentence:
    :param model:
    :param device:
    :return:
    """
    if device is None:
        device = DEVICE

    encoding = tokenizer.encode_plus(
        sentence,
        max_length=MAX_LEN,
        add_special_tokens=True,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_token_type_ids=False,
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    output = model(input_ids, attention_mask)
    # output = F.softmax(output, dim=1)
    _, pred = torch.max(output, dim=1)

    print(sentence)
    print("\nLabel: ", CLASS_NAMES[pred])


def pred_df(filtered_df, model, tokenizer, device=DEVICE):
    """
    :param tokenizer:
    :param model:
    :param device:
    :param filtered_df:
    :return:
    """
    predicted = []

    for sample in filtered_df['comments']:
        encoding = tokenizer.encode_plus(
            sample,
            max_length=MAX_LEN,
            add_special_tokens=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        output = model(input_ids, attention_mask)
        # output = F.softmax(output, dim=1)
        _, pred = torch.max(output, dim=1)

        predicted.append(pred.detach().cpu().numpy())

    filtered_df['predicted'] = predicted
    return filtered_df


