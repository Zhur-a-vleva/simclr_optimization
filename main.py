import json
import logging
from sys import argv

import torch
from tqdm import tqdm

from classes.baseline_model import Baseline
from dataset.dataset_preparation import Dataset

MODEL = "baseline"
BATCH_SIZE = 256
EPOCHS = 1000
LEARNING_RATE = 0.5
TEMPERATURE = 0.5
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    args = argv[1:]
    if len(args) < 1:
        tqdm.write(f"The parameters are set to default: model = {MODEL} batch_size = {BATCH_SIZE}, epochs = {EPOCHS}")
    elif len(args) != 3:
        tqdm.write("Please, provide all parameters or none")
        exit(1)
    else:
        m = args[0]
        b = args[1]
        e = args[2]

        if m in ["baseline"]:
            MODEL = m
        else:
            tqdm.write("Incorrect name of the model")

        try:
            b = int(b)
            if 0 < int(b):
                BATCH_SIZE = b
            else:
                tqdm.write("Incorrect batch size parameter")
        except e:
            tqdm.write(e)

        try:
            e = int(e)
            if 0 < int(e):
                EPOCHS = e
            else:
                tqdm.write("Incorrect epochs parameter")
        except e:
            tqdm.write(e)

    logger = logging.getLogger("LOGGER")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(f"log/training_log_{MODEL}.log", mode='w')
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info("Log started")
    logger.info(
        f"Constants set: MODEL={MODEL}, BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}, LEARNING_RATE={LEARNING_RATE}, TEMPERATURE={TEMPERATURE}, NUM_WORKERS={NUM_WORKERS}, DEVICE={DEVICE}")

    logger.info("Dataset download...")
    dataset = Dataset(BATCH_SIZE, NUM_WORKERS)
    dataset.prepare_data()
    logger.info("Data loaded and transformed")

    model = None
    if MODEL == "baseline":
        model = Baseline(TEMPERATURE, DEVICE, LEARNING_RATE, EPOCHS, dataset, logger)
    logger.info("Model initialized")
    model.train()
    model.load_best_model()
    model.linear_classification.evaluate()

    with open(f"metrics/metrics_{MODEL}.json", "w") as f:
        json.dump(model.metrics.metrics, f, indent=4)

    logger.info("Metrics saved")
    tqdm.write(f"Metrics saved in metrics/metrics_{MODEL}.json")