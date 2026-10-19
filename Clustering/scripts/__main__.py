import argparse
from pathlib import Path

from .data import ROOT


def main():
    parser = argparse.ArgumentParser(description="Manuscript DEC clustering")
    sub = parser.add_subparsers(dest="command", required=True)
    infer = sub.add_parser("predict", help="Reproduce results from released checkpoints")
    infer.add_argument("--workflow", choices=["precancer", "deconvolution", "all"], default="all")
    infer.add_argument("--output", type=Path, default=ROOT / "results")
    infer.add_argument("--data", type=Path)
    infer.add_argument("--checkpoint", type=Path)
    plot = sub.add_parser("plot", help="Create UMAP and PCA views from checkpoint embeddings")
    plot.add_argument("--workflow", choices=["precancer", "deconvolution", "all"], default="all")
    plot.add_argument("--output", type=Path, default=ROOT / "results" / "figures")
    plot.add_argument("--seed", type=int, default=42)
    train = sub.add_parser("train", help="Train a new model; historical checkpoints are never overwritten")
    train.add_argument("--workflow", choices=["precancer", "deconvolution"], required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--pretrain-epochs", type=int, required=True)
    train.add_argument("--epochs", type=int, required=True)
    train.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "predict":
        if args.workflow == "all" and (args.data or args.checkpoint):
            parser.error("--data/--checkpoint require one workflow")
        from .pipeline import predict
        for name in (["precancer", "deconvolution"] if args.workflow == "all" else [args.workflow]):
            frame = predict(name, args.output / name, args.data, args.checkpoint)
            print(f"{name}: {len(frame)} samples -> {args.output / name}")
    elif args.command == "plot":
        from .visualize import plot_workflow
        for name in (["precancer", "deconvolution"] if args.workflow == "all" else [args.workflow]):
            plot_workflow(name, args.output / name, args.seed)
    elif args.command == "train":
        if args.epochs < 1 or args.pretrain_epochs < 1:
            parser.error("Epoch counts must be positive")
        from .train import train_workflow
        train_workflow(args.workflow, args.output, args.pretrain_epochs, args.epochs, args.seed)


if __name__ == "__main__":
    main()
