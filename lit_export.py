import argparse
from pathlib import Path

from transformers import PreTrainedModel

from lit_module import LitModule


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        type=int,
        help="Pytorch lightning checkpoint of version to export",
        required=True,
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    lightning_logs_dir_path = Path("lightning_logs").joinpath(f"version_{args.version}")
    exports_dir_path = Path("exports").joinpath(f"version_{args.version}")

    checkpoint_file_path = next(lightning_logs_dir_path.glob("checkpoints/*.ckpt"))

    lit_module = LitModule.load_from_checkpoint(checkpoint_file_path)
    model: PreTrainedModel = lit_module.__core_module__
    model.save_pretrained(exports_dir_path)
