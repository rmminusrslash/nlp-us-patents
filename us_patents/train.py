import os
import random

import hydra
import numpy as np
import pandas as pd
import torch
import wandb
from datasets import Dataset
from dotenv import load_dotenv
from loguru import logger
from omegaconf import OmegaConf
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)

from us_patents import data
from us_patents.callbacks import SaveBestModelCallback
from us_patents.metrics import PearsonCorrelationCoefficient

load_dotenv()


def set_seeds(seed=42):
    """
    Function that sets seed for pseudo-random number generators in:
    pytorch, numpy, python.
    Makes sure to get reproducible results.
    Args:
        seed (int, optional): seed value. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"-> Global seed set to {seed}")


@hydra.main(config_name="config.yaml", config_path=".", version_base="1.1")
def simple_baseline(cfg: OmegaConf):
    """simplified version of a training script with only the bare essentials"""

    if cfg.run.debug:
        logger.warning("Running in debug mode. Weights & Biases logging is disabled\n")
        cfg.wandb.enabled = False
        cfg.trainer.report_to = "none"
        cfg.trainer.num_train_epochs = 1

    local_run = os.environ.get("USER") != "root"
    if local_run:
        cfg.trainer.fp16 = False
    else:
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

    if cfg.wandb.enabled:
        logger.info("Weights & Biases logging is enabled")
        wandb.init(
            config=cfg,
            project=cfg.wandb.project,
            group=cfg.run.experiment_name,
            name=cfg.wandb.name,
            notes=cfg.wandb.notes,
        )

    set_seeds(cfg.run.seed)

    df = pd.read_csv(f"{cfg.data.input_dir}/train.csv")

    if cfg.run.debug:
        df = df.sample(100).reset_index()

    df = data.prepare_data(df, cfg.data.cpc_scheme_xml_dir, cfg.data.cpc_title_list_dir)
    df = data.create_folds(df)

    # load pretrained model
    model_name = "bert-large-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1)

    if cfg.run.debug:
        print(model)

    # make it a regression problem (1 output)
    model.classifier = torch.nn.Linear(model.config.hidden_size, out_features=1)
    model.criterion = torch.nn.MSELoss()
    """todo: check what difference the loss made
    if loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_type == "bce":
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif loss_type == "pearson":
            self.loss_fn = CorrLoss()"""
    model.optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    def tokenize(text):
        res = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=True,
            return_token_type_ids=True,
        )

        return res["input_ids"], res["token_type_ids"], res["attention_mask"]

    df[["input_ids", "token_type_ids", "attention_mask"]] = df.apply(
        lambda x: tokenize(x.input_text), axis=1, result_type="expand"
    )

    logger.info(
        f"Tokenize inputs, example: " f"{tokenizer.decode(df.loc[0].input_ids)}"
    )
    # the model expects certain column names for input and output
    df.rename(columns={"score": "label"}, inplace=True)

    # select one of the folds for evaluation
    ds_train = Dataset.from_pandas(df[df.fold != 4])
    ds_val = Dataset.from_pandas(df[df.fold == 4])

    trainer_args = TrainingArguments(**cfg.trainer)

    trainer = Trainer(
        model,
        trainer_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tokenizer,
        compute_metrics=PearsonCorrelationCoefficient(),
        callbacks=[
            SaveBestModelCallback(metric_name=PearsonCorrelationCoefficient.name)
        ],
    )
    trainer.train()


if __name__ == "__main__":
    simple_baseline()
