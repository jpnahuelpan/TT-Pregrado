# -*- coding: utf-8 -*-
import tweepy
import config
import pandas as pd


auth = tweepy.OAuth1UserHandler(
   config.API_KEY,
   config.API_SECRET,
   config.ACCESS_TOKEN,
   config.ACCESS_TOKEN_SECRET,
)
api = tweepy.API(auth)

users = ["FalabellaAyuda", "tiendas_paris", "RipleyChile"]
for user in users:
    query = f"to:{user}"

    searched_tweets = [status for status in tweepy.Cursor(
        api.search_tweets,
        q=query,
        include_rts=False).items(450)]
    data = []
    for tweet in searched_tweets:
        data.append([tweet.user.id, tweet.user.location, tweet.created_at, tweet.text.replace('\n', ' ')])
        print(f"text: {tweet.text}\n")

    df = pd.DataFrame(
        data,
        columns=["user_id", "user_location", "created_at", "tweet_text"],
        )
    df.to_csv(f"data/raw/{user}.csv", index=False)
