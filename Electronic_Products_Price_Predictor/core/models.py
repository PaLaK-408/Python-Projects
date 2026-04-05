from django.db import models

class Prediction(models.Model):
    brand = models.CharField(max_length=200)
    category = models.CharField(max_length=200)
    rating = models.FloatField()
    discount = models.FloatField()
    sales = models.IntegerField()
    predicted_price = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.brand} - ₹{self.predicted_price}"