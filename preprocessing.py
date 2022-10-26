import re
from vncorenlp import VnCoreNLP


def rm_mark(sentence):
    sentence = sentence.replace("\n", " , ").replace(".", " , ") \
        .replace(";", " ").replace("“", " ") \
        .replace(":", " ").replace("'", " ") \
        .replace('"', " ").replace("'", " ") \
        .replace("!", " ").replace("?", " ") \
        .replace("-", " ").replace("_", " ") \
        .replace("<", " ").replace(">", " ") \
        .replace("(", " ").replace(")", " ") \
        .replace("=", " ").replace("+", " ") \
        .replace("{", " ").replace("}", " ") \
        .replace("[", " ").replace("]", " ")
    sentence = sentence.strip().lower()
    return sentence


def rm_acronyms(sentence):
    """
    Replace acronyms
    :param sentence: input sentence
    :return: sentence
    """
    sentence = sentence.replace(" mn", " mọi người ").replace(" m,n ", " mọi người ") \
        .replace(" z", " vậy ").replace("sp", " sản phẩm ") \
        .replace(" bth ", " bình thường ").replace(" r ", " rồi ") \
        .replace(" mh ", " mình ").replace(" m ", " mình ") \
        .replace(" k ", " không  ").replace(" kg ", " không ") \
        .replace(" khong", " không ").replace(" kh ", " không ") \
        .replace(" đc", " được ").replace(" dc", " được ") \
        .replace(" lun ", " luôn ").replace(" bh", " bao giờ ") \
        .replace("tl", " trả lời ").replace(" j", " gì ") \
        .replace(" siu ", " siêu ").replace(" b ", " bạn ") \
        .replace("ko", " không ").replace(" t ", " tôi ")

    return sentence.strip()


def rm_emoji(sentence):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', sentence)


def rm_url(sentence):
    url_pattern = re.compile(r'http\S+')
    return url_pattern.sub(r'', sentence)


def rm_stopwords(sentence, dictionary):
    """
    Remove words that are not in dictionary (file: tokenizer.pkl)
    :param sentence: input sentence
    :param dictionary: tokenizer
    :return: sentence
    """
    sentence = sentence.split()
    output = []
    for word in sentence:
        if word in dictionary.word_index:
            output.append(word)
    return " ".join(map(str, output))


def rm_duplicates(sentence):
    # Remove duplicates mark, ex: ,,, or ... or duplicates space
    sentence = re.sub(r',,+', ',', sentence)
    # sentence = re.sub(r'..+', '.', sentence)
    return re.sub(r'\s\s+', ' ', sentence.strip())


def standardize_data(sentence):
    filtered_sentence = rm_mark(sentence)
    filtered_sentence = rm_acronyms(filtered_sentence)
    filtered_sentence = rm_url(filtered_sentence)
    filtered_sentence = rm_emoji(filtered_sentence)
    filtered_sentence = rm_duplicates(filtered_sentence)
    return filtered_sentence


def rm_header(sentence, header, dictionary):
    """
    Remove headers that are not carry information in sentence
    :param sentence: input sentence
    :param header: header will be removed
    :param dictionary: tokenizer
    :return: sentence
    """
    sentences = str(sentence).strip(",")
    sentences = sentences.split(",")

    if len(sentences) > 2:
        if header['material'] in sentences[0]:
            del sentences[0]

        if header['color'] in sentences[0]:
            del sentences[0]

        if header['describe'] in sentences[0]:
            del sentences[0]

    i = 0
    seq_length = len(sentences)
    while i < seq_length:
        sentences[i] = rm_stopwords(sentences[i], dictionary)
        if (header['ship'] in sentences[i]) or (header['coin'] in sentences[i]):
            del sentences[i]
            seq_length = seq_length - 1
        i = i + 1

    sentences = ",".join(map(str, sentences))

    return sentences.strip(",").strip()


def preprocess_data(df):
    """
    Apply word segmenter to produce word-segmented texts before feeding to PhoBERT.
    Using RDRSegmenter from VNCoreNLP to pre-process the pre_training data.
  """
    with VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg",
                   max_heap_size='-Xmx500m') as rdrsegmenter:
        df["comments"] = df["comments"].apply(str).progress_apply(
            lambda x: ' '.join([' '.join(sent) for sent in rdrsegmenter.tokenize(x)]))

    return df
