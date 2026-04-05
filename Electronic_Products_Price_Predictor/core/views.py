from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from .models import Prediction
import os, joblib, json, numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

MODEL_PATH = os.path.join(settings.BASE_DIR, "ml_model", "model.pkl")
ENCODER_PATH = os.path.join(settings.BASE_DIR, "ml_model", "encoder.pkl")

model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODER_PATH)


def home(request):
    return render(request, "home.html")

print(encoders.keys())

def predict_page(request):
    return render(request, "predict.html")




def get_dropdown_data(request):
    return JsonResponse({
        "brands": list(encoders["brand"].classes_),
        "categories": list(encoders["categories"].classes_),
        "primaryCategories": list(encoders["primaryCategories"].classes_),
        "availability": list(encoders["prices.availability"].classes_),
        "condition": list(encoders["prices.condition"].classes_),
        "isSale": list(encoders["prices.isSale"].classes_)   
    })

def history(request):
    records = Prediction.objects.all().order_by("-created_at")
    return render(request, "history.html", {"records": records})


@csrf_exempt
def predict_api(request):

    if request.method == "POST":

        try:
            data = json.loads(request.body)

            def safe_encode(column, value):
                if column not in encoders:
                    raise ValueError(f"Encoder missing for {column}")
                if value not in encoders[column].classes_:
                    return 0
                return encoders[column].transform([value])[0]

            brand = safe_encode("brand", data.get("brand"))
            category = safe_encode("categories", data.get("category"))
            primary = safe_encode("primaryCategories", data.get("primary"))
            availability = safe_encode("prices.availability", data.get("availability"))
            condition = safe_encode("prices.condition", data.get("condition"))
            isSale = safe_encode("prices.isSale", data.get("isSale"))

            weight = float(data["weight"]) if data["weight"] else 0

            features = np.array([[ 
                brand,
                category,
                primary,
                availability,
                condition,
                isSale,
                weight
            ]])

            print("Features:", features)
            print("Expected:", model.n_features_in_)

            prediction = model.predict(features)[0]

            return JsonResponse({
                "price": round(float(prediction), 2)
            })

        except Exception as e:
            print("Prediction Error:", str(e))
            return JsonResponse({"error": str(e)}, status=500)
        

def xgboost_graph(request):
    # Load sample dataset
    data = load_iris()
    X = data.data
    y = data.target

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Train XGBoost
    model = XGBClassifier()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Feature importance
    importance = model.feature_importances_.tolist()
    features = data.feature_names

    context = {
        'accuracy': round(accuracy * 100, 2),
        'importance': json.dumps(importance),
        'features': json.dumps(features)
    }

    return render(request, 'xgboost_graph.html', context)