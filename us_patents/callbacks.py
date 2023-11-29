import numpy as np
from loguru import logger
from transformers import TrainerCallback


class SaveBestModelCallback(TrainerCallback):
    """
    Saves best model according to the competition metric and
    also logs the scores after each evaluation routine to
    custom log file
    """

    def __init__(self, metric_name: str):
        self.bestScore = 0
        self.metric_name = "eval_" + metric_name
        logger.add(
            "eval.log", filter=lambda record: record["extra"].get("task", 0) == "eval"
        )

    def on_train_begin(self, args, state, control, **kwargs):
        assert (
            args.evaluation_strategy != "no"
        ), "SaveBestModelCallback requires IntervalStrategy of steps or epoch"

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric_value = metrics.get(self.metric_name)
        eval_logger = logger.bind(task="eval")

        if metric_value > self.bestScore:
            eval_logger.info(
                f"** {self.metric_name} score improved from "
                f"{np.round(self.bestScore, 4)} to {np.round(metric_value, 4)} **"
            )
            self.bestScore = metric_value
            control.should_save = True
        else:
            eval_logger.info(
                f"{self.metric_name} score {np.round(metric_value, 4)} "
                f"(Prev. Best {np.round(self.bestScore, 4)}) "
            )
