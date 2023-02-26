# -*- coding: utf-8 -*-
import torch
import re
import pandas as pd
import numpy as np
from cleantext import clean
from transformers import pipeline


class Features:
    """
    Contiene los metodos necesario para preparar los datos,
    tanto limpieza, clasificación, codificador a bert entradas,
    tranformacion a representación de texto Bert.
    """
    def clean_texts(input_path, output_path):
        df = pd.read_csv(f"{input_path}")
        df.drop_duplicates(subset=['tweet_text'], inplace=True)
        dic = {
            "@usernames": [],
            "hashtags": [],
        }
        for row in df.iterrows():
            in_text = row[1]["tweet_text"]
            if re.findall(r"@(\w+)", in_text):
                dic["@usernames"].append(re.findall(r"@(\w+)", in_text))
            else:
                dic["@usernames"].append(float("nan"))
            if re.findall(r"#(\w+)", in_text):
                dic["hashtags"].append(re.findall(r"#(\w+)", in_text))
            else:
                dic["hashtags"].append(float("nan"))
            out_text = in_text.lower()
            # eliminación de emojis
            out_text = clean(out_text, no_emoji=True)
            # eliminación de t.co link
            out_text = re.sub(r'https?:\/\/\S*', '', out_text,
                              flags=re.MULTILINE)
            # eliminación de @acounts
            out_text = re.sub(r'@\S*', '', out_text)
            # eliminación de hashtags
            out_text = re.sub(r'#\S*', '', out_text)
            # eliminación de espacios multiples.
            out_text = re.sub(r'^ *', '', out_text)
            out_text = re.sub(r' \B', '', out_text)

            df["tweet_text"].replace([in_text], str(out_text), inplace=True)
        df["@usernames_in_tweet"] = dic["@usernames"]
        df["hashtags_in_tweet"] = dic["hashtags"]
        df.to_csv(f"{output_path}", index=False)

    def sentiment_clasificator(input_path, output_path):
        df = pd.read_csv(f"{input_path}")
        df.drop_duplicates(subset=['tweet_text'], inplace=True)
        model_path = "daveni/twitter-xlm-roberta-emotion-es"
        emotion_analysis = pipeline(
            "text-classification",
            framework="pt",
            model=model_path,
            tokenizer=model_path,
            )
        sentiment_list = []
        score_list = []
        for row in df.iterrows():
            text = str(row[1]["tweet_text"])
            sentiment_list.append(emotion_analysis(text)[0]["label"])
            score_list.append(emotion_analysis(text)[0]["score"])
        df["sentiment"] = sentiment_list
        df["score"] = score_list
        df.to_csv(f"{output_path}", index=False)

    def bert_encoder(texts, tokenizer, max_length_text=100):
        """ Genera las entradas para el modelo BERT.
        Args:
            texts:
                List: lista de textos a tranformar.
            tokenizer:
                BertTokenizer.from_pretrained(): Bertokenizer cargado con BETO.
            max_lengh_text:
                Int: Cantidad maxima de tokens que se generaran,
                se truncara en 100 en caso que sobrepase.
        Returns:
            output:
                Dict: Entradas para el modelo preentrenado BERT.
        Raises:
        """
        output = {}
        for text in texts:
            tokens = tokenizer.tokenize(text)
            tokens = ["[CLS]"] + tokens + ["[SEP]"]

            pads = max_length_text - len(tokens)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_ids += [1] * pads

            token_type_ids = [0] * max_length_text

            attention_mask = [1] * len(tokens) + [0] * pads

            if output:
                output["input_ids"] = torch.vstack((
                    output["input_ids"], torch.tensor([input_ids])))
                output["token_type_ids"] = torch.vstack((
                    output["token_type_ids"], torch.tensor([token_type_ids])))
                output["attention_mask"] = torch.vstack((
                    output["attention_mask"], torch.tensor([attention_mask])))
            else:
                output["input_ids"] = torch.tensor([input_ids])
                output["token_type_ids"] = torch.tensor([token_type_ids])
                output["attention_mask"] = torch.tensor([attention_mask])
        return output

    def max_polling(tensor):
        output = []
        for vector in tensor:
            features_extracted = []
            n = len(vector[:])
            for k in range(0, 768):
                max_element = float("-inf")
                for i in range(0, n):
                    if (vector[i][k] > max_element):
                        max_element = float(vector[i][k])
                features_extracted.append(max_element)
            output.append(np.array(features_extracted))
        return output

    def mean_polling(tensor):
        output = []
        for vector in tensor:
            features_extracted = []
            n = len(vector[:])
            for k in range(0, 768):
                sum_elements = 0
                for i in range(0, n):
                    sum_elements += float(vector[i][k])
                h = sum_elements/n
                features_extracted.append(h)
            output.append(np.array(features_extracted))
        return output

    def std_normalization(list):
        output = []
        for vector in list:
            norm1_of_vector = vector.__abs__().sum()
            vector_normalized = vector / norm1_of_vector
            output.append(vector_normalized)
        return output

    def z_score_normalization(list):
        output = []
        for vector in list:
            vector_normalized = (vector - vector.mean()) / vector.std()
            output.append(vector_normalized)
        return output

    def min_max_normalization(list, new_min=0.0, new_max=1.0):
        output = []
        for vector in list:
            div = ((vector - vector.min()) / (vector.max() - vector.min()))
            vector_normalized = (div * (new_max - new_min)) + new_min
            output.append(vector_normalized)
        return output
