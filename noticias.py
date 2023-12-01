from newsapi import NewsApiClient
from datetime import datetime, timedelta
import pandas as pd

newsapi = NewsApiClient(api_key="9aaaf2a83e8b4c59a2fac9ae1dcf58a8")

data = newsapi.get_everything(q='Cristiano Ronaldo', language='es', sort_by='relevancy', page_size=100)

articles = data['articles']

current_date = datetime.now()

for x, article in enumerate(articles):
    # Convierte la fecha de publicación a un objeto datetime
    published_date = datetime.strptime(article['publishedAt'], "%Y-%m-%dT%H:%M:%SZ")

    # Calcula la diferencia de tiempo
    time_difference = current_date - published_date

    # Filtra las noticias que tienen menos de dos días de antigüedad
    if time_difference <= timedelta(days=2):
        print(f"{x + 1}. Título: {article['title']}")
        print(f"   Enlace: {article['url']}")
        print(f"   Publicado en: {article['publishedAt']}")
        print("\n")
