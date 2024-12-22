import h5py

database_file = 'C:\Users\Tiago\Kreuzgasse Onedrive\Desktop\Tiago\Corona Aufgaben\Physik\GYPT Theorie\Database\Simulation_database.h5'  # Replace with your actual file path

# Open the HDF5 file in read mode
with h5py.File(database_file, 'r') as db:
    print("Database Structure:")
    db.visit(print)  # Print all group and dataset names recursively
