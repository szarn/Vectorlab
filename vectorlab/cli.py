"""Command-line interface.

Heavy modules are imported inside the command handlers, so logging is configured before
torch and transformers load and ``--help`` stays fast. Expected user errors are reported
here as one-line messages. Anything else is a bug and keeps its traceback.
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import yaml

from vectorlab.errors import UserError

NOISY_LOGGERS = ("torch.distributed.elastic.multiprocessing.redirects",)
RUN_HINTS = {
    "edit.json": "choose an intervention with `vectorlab edit` first",
    "eval.json": "run `vectorlab eval` first",
}


def configure_logging(verbose: bool) -> None:
    """Quiet by default; with ``verbose``, show progress logs, library warnings, and progress bars."""
    from transformers.utils import logging as hf_logging

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("vectorlab").setLevel(logging.INFO if verbose else logging.WARNING)
    logging.captureWarnings(True)
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)
    if verbose:
        hf_logging.set_verbosity_warning()
        hf_logging.enable_progress_bar()
    else:
        logging.getLogger("py.warnings").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore")
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()


def require_model(model_id: str) -> None:
    """Raise UserError if ``model_id`` is missing, gated without access, or has no adapter."""
    from huggingface_hub.errors import GatedRepoError
    from transformers import AutoConfig

    from vectorlab.adapters import check_supported

    try:
        config = AutoConfig.from_pretrained(model_id)
    except OSError as err:
        if isinstance(err, GatedRepoError) or isinstance(err.__cause__, GatedRepoError):
            raise UserError(
                f"'{model_id}' is gated on Hugging Face; accept its license at "
                f"https://huggingface.co/{model_id} and check `hf auth whoami`"
            ) from err
        raise UserError(f"cannot load model '{model_id}': {str(err).splitlines()[0]}") from err
    check_supported(config.model_type)


def require_run(run: Path, *files: str) -> Path:
    """Raise UserError unless ``run`` is a run directory containing ``files``."""
    if not (run / "run.json").is_file():
        raise UserError(f"no run at {run}; create one with `vectorlab extract`")
    for name in files:
        if not (run / name).is_file():
            raise UserError(f"{run} has no {name}; {RUN_HINTS.get(name, 're-run `vectorlab extract`')}")
    return run


def require_file(path: Path, hint: str) -> None:
    """Raise UserError with ``hint`` if ``path`` does not exist."""
    if not path.is_file():
        raise UserError(f"{path} not found; {hint}")


def cmd_extract(args):
    from vectorlab.behavior import load_behavior
    from vectorlab.extract import extract

    load_behavior(args.behavior)
    require_model(args.model)
    out, separation = extract(args.behavior, args.model, args.seed, args.batch_size)
    print("layer  cohens_d")
    for layer, d in enumerate(separation.tolist()):
        print(f"{layer:5d}  {d:8.3f}")
    print(out)


def cmd_analyze(args):
    from vectorlab.report import analyze

    require_run(args.run, "separation.json", "logit_lens.json", "projections.safetensors")
    print(analyze(args.run))


def cmd_edit(args):
    from vectorlab.edit import edit

    require_run(args.run, "directions.safetensors")
    print(json.dumps(edit(args.run, args.layer, args.mode, args.alpha, args.range, args.weights)))


def cmd_eval(args):
    from vectorlab.evaluate import NEUTRAL, evaluate

    require_run(args.run, "directions.safetensors", "edit.json")
    require_file(NEUTRAL, "create it with `python scripts/fetch_refusal.py`")
    result = evaluate(args.run, args.n, args.batch_size, args.judge)
    print(f"{result['metric']:>10}  before {result['before']:.3f}  after {result['after']:.3f}  (n={result['n']})")
    print(f"{'kl':>10}  {result['kl']:.4f}")


def build_parser() -> argparse.ArgumentParser:
    """The ``vectorlab`` argument parser with one subcommand per command."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true", help="log progress and library warnings")

    parser = argparse.ArgumentParser(prog="vectorlab", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(required=True, metavar="command")

    p = sub.add_parser("extract", parents=[common], help="capture activations and compute directions")
    p.add_argument("behavior", help="behavior YAML spec")
    p.add_argument("--model", required=True, help="Hugging Face model id")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=16)
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("analyze", parents=[common], help="write report.html for a run")
    p.add_argument("run", type=Path)
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("edit", parents=[common], help="choose the intervention to evaluate")
    p.add_argument("run", type=Path)
    p.add_argument("--layer", type=int, required=True, help="layer whose direction to use")
    p.add_argument("--mode", choices=["ablate", "add"], help="defaults to the behavior's intervention")
    p.add_argument("--alpha", type=float, help="ablation scale (default 1.0) or addition coefficient")
    p.add_argument("--range", help="START:END layers to intervene on")
    p.add_argument("--weights", action="store_true", help="bake the edit into the weights")
    p.set_defaults(func=cmd_edit)

    p = sub.add_parser("eval", parents=[common], help="measure the edit's behavior metric and KL")
    p.add_argument("run", type=Path)
    p.add_argument("--n", type=int, help="evaluate only the first N held-out prompts")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--judge", default="Qwen/Qwen3-4B-Instruct-2507", help="judge model for judge metrics")
    p.set_defaults(func=cmd_eval)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``vectorlab`` console script."""
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    try:
        args.func(args)
    except UserError as err:
        sys.exit(f"vectorlab: error: {err}")
    except (yaml.YAMLError, json.JSONDecodeError) as err:
        sys.exit(f"vectorlab: error: could not parse input file: {err}")
