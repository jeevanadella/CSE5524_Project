'''
Sample predictive model.
The ingestion program will call `predict` to get a prediction for each test image and then save the predictions for scoring. The following two methods are required:
- predict: uses the model to perform predictions.
- load: reloads the model.
'''
import torch
import os
from open_clip import create_model_and_transforms
import torch.nn as nn

def get_bioclip():
    """function that returns frozen bioclip model

    model: bioclip
    """
    # bioclip = create_model("hf-hub:imageomics/bioclip-2", output_dict=True, require_pretrained=True).cuda()
    bioclip, _, preprocess = create_model_and_transforms(
        "hf-hub:imageomics/bioclip-2", output_dict=True, require_pretrained=True
    )
    bioclip = bioclip.cuda()
    return bioclip, preprocess


import torch.nn as nn

class BioClip2_DeepRegressor(nn.Module):
    def __init__(
        self,
        bioclip,
        num_features=768,
        hidden_size_begin=512,
        hidden_layer_decrease_factor=4,
        num_outputs=3,
    ):
        super().__init__()
        # regressor linear layer
        self.bioclip = bioclip
        self.regressor = nn.Sequential(
            # 768 = num features output from bioclip
            nn.Linear(in_features=num_features, out_features=hidden_size_begin),
            nn.GELU(),
            nn.Linear(
                in_features=hidden_size_begin,
                out_features=int(hidden_size_begin / hidden_layer_decrease_factor),
            ),
            nn.GELU(),
            nn.Linear(
                in_features=int(hidden_size_begin / hidden_layer_decrease_factor),
                out_features=int(hidden_size_begin / hidden_layer_decrease_factor**2),
            ),
            nn.GELU(),
            nn.Linear(
                in_features=int(hidden_size_begin / hidden_layer_decrease_factor**2),
                out_features=num_outputs,
            ),
        )

    def forward(self, x):
        return self.regressor(self.bioclip(x)["image_features"])

class Model:
    def __init__(self):
        # model will be called from the load() method
        self.model = None
        self.transforms = None

    def load(self):
        bioclip,_, transforms = create_model_and_transforms( "hf-hub:imageomics/bioclip-2", output_dict=True, require_pretrained=True)
        self.transforms = transforms
        model_path = os.path.join(os.path.dirname(__file__), "model.pth")
        self.model = BioClip2_DeepRegressor(bioclip=bioclip)
        self.model.load_state_dict(torch.load(model_path))
            

    def predict(self, datapoints):
        images = [entry['img'] for entry in datapoints]
        tensor_images = torch.stack([self.transforms(image) for image in images])
        #model outputs 30d,1y,2y
        outputs = self.model(tensor_images)
        mu = torch.mean(outputs, dim=0)
        sigma = torch.std(outputs,dim=0)
        return {
        'SPEI_30d': {
            'mu': mu[0].item(),
            'sigma': sigma[0].item()
        },
        'SPEI_1y': {
            'mu': mu[1].item(),
            'sigma': sigma[1].item()
        },
        'SPEI_2y': {
            'mu': mu[2].item(),
            'sigma': sigma[2].item()
        }
}   
    
        
        
