import logging
import subprocess
from pathlib import Path

import click

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def create_input(
    template_path: Path,
    output_path: Path,
    data_path: Path,
    src_train_path: Path,
    tgt_train_path: Path,
    src_val_path: Path,
    tgt_val_path: Path,
    log_path: Path,
) -> Path:
    # Create nmt yaml
    with template_path.open("r") as f:
        template = f.read()

    src_vocab_path = output_path / "data" / "vocab" / "vocab.src"
    tgt_vocab_path = output_path / "data" / "vocab" / "vocab.tgt"

    save_model_path = output_path / "model"

    input_file = template.format(
        data_path,
        src_vocab_path,
        tgt_vocab_path,
        src_train_path,
        tgt_train_path,
        src_val_path,
        tgt_val_path,
        log_path,
        save_model_path,
    )

    input_file_path = output_path / "input.yaml"
    with input_file_path.open("w") as f:
        f.write(input_file)

    return input_file_path


def gen_vocab(log_path: Path, input_file_path: Path) -> None:
    # Create vocab
    with (log_path / "vocab.log").open("w") as out:
        subprocess.call(
            ["onmt_build_vocab", "-config", input_file_path, "-n_sample", "-1"],
            stdout=out,
            stderr=out,
        )


@click.command()
@click.option("--template_path", type=Path, required=True, help="Path to the config template")
@click.option("--data_folder", type=Path, required=True, help="Data folder")
def main(template_path: Path, data_folder: Path):
    logging.basicConfig(level="INFO")

    log_path = data_folder / "logs"
    log_path.mkdir(parents=True, exist_ok=True)

    data_path = data_folder / "data"

    src_train_path = data_path / "src-train.txt"
    tgt_train_path = data_path / "tgt-train.txt"
    src_val_path = data_path / "src-val.txt"
    tgt_val_path = data_path / "tgt-val.txt"

    # Create input yaml
    logger.info("Creating input...")
    input_file_path = create_input(
        template_path,
        data_folder,
        data_path,
        src_train_path,
        tgt_train_path,
        src_val_path,
        tgt_val_path,
        log_path,
    )

    # Create vocab files
    logger.info("Creating vocab...")
    gen_vocab(log_path, input_file_path)

    input_file_path = data_folder / "input.yaml"

    # Start trainig
    logger.info("Starting training...")
    train_logs_path = log_path / "train"
    out_train = train_logs_path / "out.txt"
    err_train = train_logs_path / "err.txt"

    train_logs_path.mkdir(parents=True, exist_ok=True)

    with out_train.open("w") as out_file, err_train.open("w") as err_file:
        subprocess.call(
            ["onmt_train", "-config", input_file_path], stdout=out_file, stderr=err_file
        )


if __name__ == "__main__":
    main()
