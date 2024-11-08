import argparse
import torch
import datetime
import json
import yaml
import os

from main_model import Score_Based_Diffusion
from dataset_forecasting import get_dataloader, get_test_dataloader
from utils import train, evaluate

import torch
import pickle



parser = argparse.ArgumentParser(description="SHD")
parser.add_argument("--config", type=str, default="base_queueing.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=1)

args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/queueing_" + current_time + "/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

train_loader, valid_loader, scaler, mean_scaler = get_dataloader(
    device= args.device,
    batch_size=config["train"]["batch_size"],
    config=config,
)
test_loader, scaler, mean_scaler = get_test_dataloader(
    device= args.device,
    batch_size=config["test"]["batch_size"],
    config=config,
)

model = Score_Based_Diffusion(config, args.device, config["data"]["target_dim"]).to(args.device)

if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

evaluate(
    model,
    test_loader,
    nsample=args.nsample,
    scaler=scaler,
    mean_scaler=mean_scaler,
    foldername=foldername,
    normalize = config["data"]["normalize_method"],
)

datafolder = foldername.split("/")[-2] # set the folder name
nsample = 1 # number of generated sample

path = './save/'+datafolder+"/generated_outputs_nsample_" + str(nsample) + '.pk' 
with open(path, 'rb') as f:
    samples,all_target,all_evalpoint,all_observed,all_observed_time,scaler,mean_scaler = pickle.load( f)