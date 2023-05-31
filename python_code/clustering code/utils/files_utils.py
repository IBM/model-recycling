import os


def get_results_model_dir_for_iteration(outdir, i):
    return os.path.join(get_root_model_dir(outdir), f"train_{i}")


def get_root_model_dir(outdir):
    return os.path.join(outdir, "models")