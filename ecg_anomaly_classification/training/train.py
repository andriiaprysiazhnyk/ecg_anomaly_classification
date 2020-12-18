import os
import yaml
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from ecg_anomaly_classification.data_api.data_api import load_train_set, load_validation_set
from ecg_anomaly_classification.data_api.dataset import ECGDataset
from ecg_anomaly_classification.transforms.transforms import get_transform, get_output_dim
from ecg_anomaly_classification.training.models import get_model
from ecg_anomaly_classification.training.trainer import Trainer


if __name__ == "__main__":
    # read config
    with open("config.yaml") as config_file:
        config = yaml.full_load(config_file)

    # create folder for logs
    experiment_name = f"{config['model']['type']}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    config["log_dir"] = os.path.join(config["log_dir"], experiment_name)
    os.makedirs(config["log_dir"])

    # create summary-writer
    summary_writer = SummaryWriter(config["log_dir"])

    # copy config into logs dir
    with open(os.path.join(config["log_dir"], "config.yaml"), "w") as config_copy:
        yaml.dump(config, config_copy)

    # create data transforms
    transform = get_transform(config["transform"]) if "transform" in config else None

    # create data-loaders
    ptb_path = config["ptb_path"]
    train_df, val_df = load_train_set(ptb_path), load_validation_set(ptb_path)
    train_ds, val_ds = ECGDataset(ptb_path, train_df, transform), ECGDataset(ptb_path, val_df, transform)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=config["batch_size"], shuffle=True)

    # create model
    input_size = get_output_dim(config["transform"]) if "transform" in config else 12
    model = get_model(config["model"], input_size)

    # configure device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # train model
    trainer = Trainer(config, train_dl, val_dl, model, summary_writer, device)
    trainer.train()
