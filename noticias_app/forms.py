# En noticias_app/forms.py
from django import forms

class NewsForm(forms.Form):
    theme = forms.CharField(label='Tema de las noticias', max_length=100)
