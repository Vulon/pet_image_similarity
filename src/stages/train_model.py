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
from datetime import datetime as dt
import json
import shutil
from src.core.metrics import cos, create_compute_metrics_function


if __name__ == "__main__":
    project_root = os.environ["DVC_ROOT"]
    params = dvc.api.params_show()
    config = get_config(params)

    sequence = create_train_sequence(config.augmentations)

    train_dataset = ImageDataset(os.path.join(project_root, config.data.train_h5_file), sequence, config.data.feature_extractor, True)
    val_dataset = ImageDataset(os.path.join(project_root, config.data.val_h5_file), sequence, config.data.feature_extractor, False)
    test_dataset = ImageDataset(os.path.join(project_root, config.data.test_h5_file), sequence, config.data.feature_extractor, False)

    if config.trainer.loss_function == "cos":
        loss_function = cos
    elif config.trainer.loss_function == "mse":
        loss_function = torch.nn.MSELoss()

    model = ResNet(config.model.output_vector_size, TripletLoss(config.model.triplet_loss_alpha, loss_function))
    tensorboard_logdir = os.path.join(project_root, config.trainer.output_folder, "tensorboard", f"{config.trainer.experiment_name}_{dt.now().strftime('%Y_%m_%d__%H_%M')}")

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
        logging_strategy = "epoch",
        logging_dir = tensorboard_logdir
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=create_compute_metrics_function()
    )
    trainer.train( config.trainer.trainer_checkpoint if config.trainer.trainer_checkpoint else None )

    shutil.copytree(tensorboard_logdir, os.path.join(project_root, config.trainer.tensorboard_log))

    val_metrics = trainer.evaluate(val_dataset)

    with open("output/val_metrics.json", 'w') as file:
        json.dump(val_metrics, file)
    if config.trainer.compute_test_metrics:
        test_metrics = trainer.evaluate(test_dataset)
        with open("output/test_metrics.json", 'w') as file:
            json.dump(test_metrics, file)

    trainer.save_model( os.path.join(project_root, config.trainer.output_folder, "model") )

