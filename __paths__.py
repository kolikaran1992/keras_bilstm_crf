from pathlib import Path
path_obj = Path(__file__).absolute().parent
path_to_logs = path_obj.joinpath('logs')