from django.urls import path
from . import views

app_name = 'image_search'

urlpatterns = [
    # API endpoints
    path('upload/', views.upload_and_index_image, name='upload_image'),
    path('search/', views.search_similar_images, name='search_images'),
    path('duplicates/', views.find_duplicates, name='find_duplicates'),
]