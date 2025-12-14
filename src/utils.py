import yaml
import logging
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
)
