from pathlib import Path
import pickle


def load_obj(name: Path):
    """
        Generic function to load a bin obj with pickle protocol

    Input :
        - name (str) : path to where the obj is
    Output :
        - (python object) : loaded object

    Possible mistakes :
        - Wrong path
    """
    with open(name.as_posix(), "rb") as f:
        return pickle.load(f)



def save_obj(obj, name: Path):
    """
        Generic function to save an obj with pickle protocol

    Input :
        - obj (python var) : object to be saved in binary format
        - name (str) : path to where the obj shoud be saved

    Possible mistakes :
        - Permissions denied, restart notebook from an admin shell
        - Folders aren't callable, change your folders
    """
    with open(name.as_posix(), "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)