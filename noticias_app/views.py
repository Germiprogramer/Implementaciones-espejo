from django.shortcuts import render
from datetime import datetime, timedelta
from newsapi import NewsApiClient  # Asegúrate de que newsapi esté instalado

def noticias(request):
    newsapi = NewsApiClient(api_key="9aaaf2a83e8b4c59a2fac9ae1dcf58a8")

    # Realiza la solicitud a la API
    data = newsapi.get_everything(q='Madrid', language='es', sort_by='relevancy', page_size=100)

    # Obtiene la lista de artículos
    articles = data['articles']

    # Obtiene la fecha actual
    current_date = datetime.now()

    # Filtra las noticias que tienen menos de dos días de antigüedad
    filtered_news = [article for article in articles if
                     (current_date - datetime.strptime(article['publishedAt'], "%Y-%m-%dT%H:%M:%SZ")) > timedelta(days=2)]

    context = {'filtered_news': filtered_news}
    return render(request, 'noticias_app/noticias.html', context)
