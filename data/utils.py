import os


def get_config_data_path() -> str:
    dir_name = os.path.normpath(
        os.path.join(os.path.dirname(__file__), os.pardir, "config", "data")
    )
    return dir_name


def write_to_config_data(file_name: str):
    dir_name = get_config_data_path()
    data_file = os.path.join(dir_name, file_name + ".py")
    text = (
        """# dataset folder name and this file should be the same\nfile_name = '"""
        + file_name
        + """'\ndataset_dir = 'data/' + file_name"""
    )

    # write text to a file that doesn't exist with variables
    # with data information to extract
    if not os.path.isfile(data_file):
        open(data_file, "w").write(text)
    else:
        return
