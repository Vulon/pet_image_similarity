import os
import sys

import dvc.api
import numpy as np
import torch.nn
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments

sys.path.append(os.environ["DVC_ROOT"])
import json
import shutil
from datetime import datetime as dt

from src.config import get_config_from_dvc
from src.core.dataset import ImageDataset
from src.core.image_processing import create_feature_extractor, create_train_sequence
from src.core.metrics import (
    TripletLoss,
    create_compute_metrics_function,
    get_base_loss_function,
)
from src.core.model import create_train_model


def save_output_files(
    tensorboard_run_folder: str,
    last_tensorboard_logs_folder: str,
    trainer: Trainer,
    model_folder: str,
    metrics_folder: str,
    val_metrics: dict,
    test_metrics: dict,
):
    shutil.copytree(tensorboard_run_folder, last_tensorboard_logs_folder)
    trainer.save_model(model_folder)

    if val_metrics is not None:
        with open(os.path.join(metrics_folder, "val_metrics.json"), "w") as file:
            json.dump(val_metrics, file)

    if test_metrics is not None:
        with open(os.path.join(metrics_folder, "test_metrics.json"), "w") as file:
            json.dump(test_metrics, file)


if __name__ == "__main__":
    project_root = os.environ["DVC_ROOT"]
    config = get_config_from_dvc()

    sequence = create_train_sequence(config.augmentations, config.random_seed)
    feature_extractor = create_feature_extractor(config.data.feature_extractor)

    train_dataset = ImageDataset(
        os.path.join(project_root, config.data.train_h5_file),
        sequence,
        feature_extractor,
        True,
    )
    val_dataset = ImageDataset(
        os.path.join(project_root, config.data.val_h5_file),
        sequence,
        feature_extractor,
        False,
    )
    test_dataset = ImageDataset(
        os.path.join(project_root, config.data.test_h5_file),
        sequence,
        feature_extractor,
        False,
    )
    loss_function = get_base_loss_function(config.trainer.loss_function)

    model = create_train_model(
        config.model.pretrained_model_name,
        config.model.output_vector_size,
        TripletLoss(config.model.triplet_loss_alpha, loss_function),
    )

    tensorboard_logdir = os.path.join(
        project_root,
        config.trainer.output_folder,
        "tensorboard",
        f"{config.trainer.experiment_name}_{dt.now().strftime('%Y_%m_%d__%H_%M')}",
    )

    args = TrainingArguments(
        output_dir=os.path.join(project_root, config.trainer.output_folder),
        evaluation_strategy="epoch",
        eval_steps=config.trainer.eval_steps,
        per_device_train_batch_size=config.trainer.train_batch_size,
        per_device_eval_batch_size=config.trainer.eval_steps,
        num_train_epochs=config.trainer.epochs,
        seed=config.random_seed,
        learning_rate=config.trainer.learning_rate,
        weight_decay=config.trainer.weight_decay,
        save_steps=config.trainer.save_steps,
        fp16=config.trainer.fp16,
        gradient_accumulation_steps=config.trainer.gradient_accumulation_steps,
        eval_accumulation_steps=config.trainer.eval_accumulation_steps,
        logging_strategy="epoch",
        logging_dir=tensorboard_logdir,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=create_compute_metrics_function(),
    )
    trainer.train(
        config.trainer.trainer_checkpoint if config.trainer.trainer_checkpoint else None
    )
    val_metrics = trainer.evaluate(val_dataset)
    if config.trainer.compute_test_metrics:
        test_metrics = trainer.evaluate(test_dataset)
    else:
        test_metrics = None

    save_output_files(
        tensorboard_logdir,
        os.path.join(project_root, config.trainer.tensorboard_log),
        trainer,
        os.path.join(project_root, config.trainer.output_folder, "model"),
        os.path.join(project_root, config.trainer.output_folder),
        val_metrics,
        test_metrics,
    )
