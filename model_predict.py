import pickle
import torch
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}")

with open("model/train_dataset.pkl", "rb") as f:
    train_dataset = pickle.load(f)

class WeightedQuantileLoss(QuantileLoss):
    def __init__(self, zero_weight=1.0, non_zero_weight=10.0, **kwargs):
        super().__init__(**kwargs)
        self.zero_weight = zero_weight
        self.non_zero_weight = non_zero_weight

    def loss(self, y_pred, y_actual):
        base_loss = super().loss(y_pred, y_actual)
        weights = torch.where(y_actual == 0, self.zero_weight, self.non_zero_weight)
        
        # Expand weights to match the shape of base_loss
        weights = weights.unsqueeze(-1).expand_as(base_loss)
        
        return (base_loss * weights).mean()

# Define the model
# Use the weighted loss in the model definition
best_tft = TemporalFusionTransformer.from_dataset(
    train_dataset,
    learning_rate=1e-3,
    hidden_size=64,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=32,
   # output_size=7,  # number of quantiles
    loss=WeightedQuantileLoss(zero_weight=1.0, non_zero_weight=10.0),  # Adjust weights as needed
    log_interval=10,
    reduce_on_plateau_patience=10
)

best_model_path = "model/best-checkpoint.ckpt"

state_dict = torch.load(best_model_path, map_location=torch.device('cpu'))
best_tft.load_state_dict(state_dict['state_dict'], strict=False)
# Load the model on CPU
# best_tft = TemporalFusionTransformer.load_from_checkpoint(
#     best_model_path, 
#     map_location=torch.device('cpu'),  # Forces everything to CPU
#     strict=False
# )

validation_dataset = TimeSeriesDataSet.from_dataset(
    train_dataset,
    #@ TODO get dataset,
    predict=True,
    stop_randomization=True,
)

batch_size = 64
val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=4)

if next(best_tft.parameters()).is_cuda:
    best_tft = best_tft.cpu()

predictions = best_tft.predict(val_dataloader)