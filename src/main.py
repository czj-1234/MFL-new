# ============================================================
# Main Entry
# ============================================================

from src.config import (
    ExperimentArgs,
    parse_cli_args,
    load_cli_config,
)
from src.runner import (
    run_one_experiment,
    run_all_experiments,
)


def main():
    cli_args = parse_cli_args()
    cfg = load_cli_config(cli_args)

    if cli_args.run_all:
        run_all_experiments(cfg, cli_args)
    else:
        args = ExperimentArgs(
            cfg,
            setting_name=cli_args.setting,
            association=cli_args.association,
            rounds=cli_args.rounds,
            samples_per_client=cli_args.samples_per_client,
            output_root=cli_args.output_root,
        )

        run_one_experiment(args)


if __name__ == "__main__":
    main()