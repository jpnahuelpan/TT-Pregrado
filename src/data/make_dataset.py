# -*- coding: utf-8 -*-
import tweepy
import pandas as pd
from datetime import date


class Data:
    """
        Contiene todos los metodos necesarios para recolectar datos
        desde twitter a traves la API twitter usando la libreria Tweepy.
    """
    def __init__(self, api):
        self.api = api

    def get_tweets(self, users, output_path):
        today = date.today()
        for user in users:
            query = f"to:{user}"

            searched_tweets = [status for status in tweepy.Cursor(
                self.api.search_tweets,
                q=query,
                tweet_mode="extended",
                truncated=False,
                include_rts=False).items(450)]
            data = []
            for tweet in searched_tweets:
                data.append([
                    tweet.user.id,
                    tweet.user.location,
                    tweet.created_at,
                    tweet.full_text.replace('\n', ' '),
                    ])

            df = pd.DataFrame(
                data,
                columns=[
                    "user_id",
                    "user_location",
                    "created_at",
                    "tweet_text"],
                )
            df.to_csv(f"{output_path}{user}_{today}.csv", index=False)
