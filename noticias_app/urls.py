# En noticias_app/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.noticias),
]

