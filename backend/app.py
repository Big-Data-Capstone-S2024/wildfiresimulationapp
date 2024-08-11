from flask import Flask, jsonify, request
import torch
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from dotenv import load_dotenv
import os

# Custom WeightedQuantileLoss class
class WeightedQuantileLoss(QuantileLoss):
    def __init__(self, zero_weight=1.0, non_zero_weight=10.0, **kwargs):
        super().__init__(**kwargs)
        self.zero_weight = zero_weight
        self.non_zero_weight = non_zero_weight

    def loss(self, y_pred, y_actual):
        base_loss = super().loss(y_pred, y_actual)
        weights = torch.where(y_actual == 0, self.zero_weight, self.non_zero_weight)
        weights = weights.unsqueeze(-1).expand_as(base_loss)
        return (base_loss * weights).mean()

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Load the model checkpoint
model_path = "./model/best-checkpoint.ckpt"
model = TemporalFusionTransformer.load_from_checkpoint(model_path)
model.eval()  

# Access the environment variables
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_ANON_KEY')
supabase_headers = {
    "apikey": supabase_key,
    "Authorization": f"Bearer {supabase_key}",
    "Content-Type": "application/json"
}

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Wildfire Simulation API"}), 200

@app.route('/api/users')
def get_users():
    try:
        response = requests.get(f"{supabase_url}/auth/v1/users", headers=supabase_headers)
        response.raise_for_status()
        return jsonify(response.json())
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    selected_date = pd.to_datetime(data['date'])
    location = data['location']

    # Load historical data from CSV
    historical_data = pd.read_csv('/historical_07_12.csv')
    
    # Filter data for the selected location and date range
    historical_data['date'] = pd.to_datetime(historical_data[['year', 'month', 'day']])
    historical_data = historical_data[(historical_data['date'] <= selected_date) & 
                                      (historical_data['date'] > selected_date - pd.Timedelta(days=30))]

    # Prepare data for the model
    dataset = TimeSeriesDataSet.from_dataset(
        original_dataset=model.hparams.dataset,
        data=historical_data,
        predict=True,
        stop_randomization=True
    )

    dataloader = dataset.to_dataloader(train=False, batch_size=64)
    
    # Make predictions
    with torch.no_grad():
        predictions = model.predict(dataloader)

    # Prepare the response
    response = []
    for i, prediction in enumerate(predictions):
        response.append({
            "date": (selected_date + pd.Timedelta(days=i)).strftime('%Y-%m-%d'),
            "location": location,
            "cfb": prediction.item()
        })

    return jsonify(response), 200

if __name__ == '__main__':
    app.run(debug=True)
