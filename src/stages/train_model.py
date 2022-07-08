import torch.nn
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader
import numpy as np
import sys, os
import dvc.api
sys.path.append(os.environ["DVC_ROOT"])
from src.config import get_config
from src.core.image_processing import create_train_sequence
from src.core.dataset import ImageDataset
from src.core.model import SwinModel, TripletLoss, ResNet


def cos(a: torch.Tensor, b: torch.Tensor):
    loss = torch.nn.CosineSimilarity()(a, b)
    loss = torch.mean(torch.abs(loss))
    return loss


def create_compute_metrics_function():


    def compute_metrics(evalPrediction):
        positive_vector, anchor_vector, negative_vector = evalPrediction.predictions

        positive_mse = (positive_vector - anchor_vector)** 2
        negative_mse = (negative_vector - anchor_vector)** 2
        mask = negative_mse > positive_mse
        return {
            "accuracy": np.mean(mask), "positive mse" : np.mean(positive_mse), "negative_mse": np.mean(negative_mse),
            "positive cos" : cos(torch.tensor(positive_mse).float(), torch.tensor(anchor_vector).float()),
            "negative cos" : cos(torch.tensor(anchor_vector).float(), torch.tensor(negative_vector).float())
        }

    return compute_metrics


if __name__ == "__main__":
    project_root = os.environ["DVC_ROOT"]
    params = dvc.api.params_show()
    config = get_config(params)

    sequence = create_train_sequence(config.augmentations)

    train_dataset = ImageDataset(os.path.join(project_root, config.data.train_h5_file), sequence, config.data.feature_extractor, True)
    val_dataset = ImageDataset(os.path.join(project_root, config.data.val_h5_file), sequence, config.data.feature_extractor, False)

    if config.trainer.loss_function == "cos":
        loss_function = cos
    elif config.trainer.loss_function == "mse":
        loss_function = torch.nn.MSELoss()

    model = ResNet(config.model.output_vector_size, TripletLoss(config.model.triplet_loss_alpha, loss_function))

    args = TrainingArguments(
        output_dir= os.path.join(project_root, config.trainer.output_folder) ,
        evaluation_strategy='epoch',
        eval_steps=config.trainer.eval_steps,
        per_device_train_batch_size=config.trainer.train_batch_size,
        per_device_eval_batch_size=config.trainer.eval_steps,
        num_train_epochs=config.trainer.epochs,
        seed=config.random_seed,
        learning_rate = config.trainer.learning_rate,
        weight_decay = config.trainer.weight_decay,
        save_steps = config.trainer.save_steps,
        fp16 = config.trainer.fp16,
        gradient_accumulation_steps = config.trainer.gradient_accumulation_steps,
        eval_accumulation_steps = config.trainer.eval_accumulation_steps,
        logging_strategy = "epoch"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=create_compute_metrics_function()
    )
    trainer.train( config.trainer.trainer_checkpoint if config.trainer.trainer_checkpoint else None )

    trainer.save_model( os.path.join(project_root, config.trainer.output_folder, "model") )

