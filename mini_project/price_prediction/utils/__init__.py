import pathlib

def get_app_root():
    return pathlib.Path(__file__).parent.parent.parent.parent.resolve()
