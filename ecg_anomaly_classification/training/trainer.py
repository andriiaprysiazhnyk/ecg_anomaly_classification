import os
import tqdm
import torch
import torch.optim as optim

from ecg_anomaly_classification.training.metrics import get_metric


class Trainer:
    def __init__(self, config, train_dl, val_dl, model, summary_writer, device):
        self.config = config
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.model = model
        self.summary_writer = summary_writer
        self.device = device
        self.criteria = torch.nn.BCEWithLogitsLoss()
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        self.metrics = {metric_name: get_metric(metric_name) for metric_name in self.config["metrics"]}

    def _get_optimizer(self):
        optimizer_config = self.config["optimizer"]

        if optimizer_config["name"] == "adam":
            return optim.Adam(self.model.parameters(),
                              optimizer_config["lr"],
                              weight_decay=optimizer_config.get("weight_decay", 0))

        raise TypeError(f"Unknown optimizer name: {optimizer_config['name']}")

    def _get_scheduler(self):
        scheduler_config = self.config["scheduler"]

        if scheduler_config["name"] == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                        mode="max",
                                                        factor=scheduler_config["factor"],
                                                        patience=scheduler_config["patience"])
        elif scheduler_config["name"] == "step":
            return optim.lr_scheduler.StepLR(self.optimizer,
                                             step_size=scheduler_config["step_size"],
                                             gamma=scheduler_config["gamma"])

        raise TypeError(f"Unknown scheduler type: {scheduler_config['name']}")

    def train(self):
        self.model.to(self.device)
        best_metric = -float("inf")

        for epoch in range(1, self.config["num_epochs"] + 1):
            train_loss = self._run_epoch(epoch)
            val_loss, metrics = self._validate()

            self._write_to_tensor_board(epoch, train_loss, val_loss, metrics)

            if metrics["accuracy"] > best_metric:
                self._save_checkpoints()
                best_metric = metrics["accuracy"]

            if self.config["scheduler"] == "plateau":
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            print(f"\nEpoch: {epoch}; train loss = {train_loss}; validation loss = {val_loss}")

    def _run_epoch(self, epoch):
        self.model.train()

        lr = self.optimizer.param_groups[0]["lr"]
        status_bar = tqdm.tqdm(total=len(self.train_dl))
        status_bar.set_description(f"Epoch {epoch}, lr {lr}")

        losses = []
        for x, y in self.train_dl:
            x, y = x.to(self.device), y.float().to(self.device)
            self.model.zero_grad()
            logits = self.model(x).squeeze()
            loss = self.criteria(logits, y)
            loss.backward()
            self.optimizer.step()

            status_bar.update()
            status_bar.set_postfix(loss=loss.item())
            losses.append(loss.item())

        status_bar.close()
        return sum(losses) / len(losses)

    def _validate(self):
        self.model.eval()

        status_bar = tqdm.tqdm(total=len(self.val_dl))

        losses, logits_total, y_total = [], torch.empty(0, dtype=torch.float32), torch.empty(0, dtype=torch.long)
        with torch.no_grad():
            for x, y in self.val_dl:
                logits = self.model(x.to(self.device)).squeeze()
                loss = self.criteria(logits, y.float().to(self.device))
                losses.append(loss.item())

                logits_total = torch.cat((logits_total, logits.cpu()))
                y_total = torch.cat((y_total, y))

                status_bar.update()
                status_bar.set_postfix(loss=loss.item())

        status_bar.close()
        return sum(losses) / len(losses), \
               {metric_name: metric(y_total, logits_total) for metric_name, metric in self.metrics.items()}

    def _save_checkpoints(self):
        torch.save(self.model.extended_state_dict(), os.path.join(self.config["log_dir"], "model.pth"))

    def _write_to_tensor_board(self, epoch, train_loss, val_loss, metrics):
        self.summary_writer.add_scalar(tag="Train loss", scalar_value=train_loss, global_step=epoch)
        self.summary_writer.add_scalar(tag="Validation loss", scalar_value=val_loss, global_step=epoch)

        for metric_name, metric_value in metrics.items():
            self.summary_writer.add_scalar(tag=metric_name, scalar_value=metric_value, global_step=epoch)
