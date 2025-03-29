import h5py


def list_h5_contents(filename):
    """
    Lists the content of an HDF5 file, including groups and datasets.

    Parameters:
        filename (str): Path to the HDF5 file.

    Returns:
        None
    """

    def visit_fn(name, obj):
        # Check if it's a group or a dataset
        if isinstance(obj, h5py.Group):
            print(f"Group: {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}, Shape: {obj.shape}, Dtype: {obj.dtype}")

    # Open the HDF5 file in read mode
    with h5py.File(filename, "r") as h5_file:
        print(f"Contents of HDF5 file: {filename}")
        h5_file.visititems(visit_fn)


# Example usage
list_h5_contents(
    "/mnt/c/Users/Tiago/Kreuzgasse Onedrive/Desktop/Tiago/Corona Aufgaben/Physik/GYPT Theorie/Database/Simulation_database.h5"
)
