import hydra
import pandas as pd
import torch
from datasets import Dataset
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


@hydra.main(config_name="config.yaml")
def simple_baseline(cfg: OmegaConf):
    """simplified version of a training script with only the bare essentials"""
    df = pd.read_csv(f"{cfg.data.input_dir}/train.csv")

    if cfg.debug:
        df = df.sample(100).reset_index()

    df = data.prepare_data(df, cfg.data.cpc_scheme_xml_dir, cfg.data.cpc_title_list_dir)
    df = data.create_folds(df)

    # load pretrained model
    model_name = "bert-large-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    print(tokenizer)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1)

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
        f"Tokenize inputs, example: {df.loc[0].input_ids}: "
        f"{tokenizer.decode(df.loc[0].input_ids)}"
    )
    # the model expects certain column names for input and output
    df.rename(columns={"score": "label"}, inplace=True)

    # select one of the folds for evaluation
    ds_train = Dataset.from_pandas(df[df.fold != 4])
    ds_val = Dataset.from_pandas(df[df.fold == 4])

    trainer_args = TrainingArguments(**cfg.bert)

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
