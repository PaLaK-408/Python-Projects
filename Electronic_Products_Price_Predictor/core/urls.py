from django.urls import path
from . import views

urlpatterns = [
    path('', views.home),
    path('predict/', views.predict_page),
    path('api/predict/', views.predict_api),
    path('history/', views.history),
    path('api/dropdowns/', views.get_dropdown_data),
    path('xgboost/', views.xgboost_graph, name='xgboost_graph'),
]