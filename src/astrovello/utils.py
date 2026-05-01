import psutil
import os

# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- Utility Functions -----------------------------------------------------

def get_filters(file_list, start, position):
    """Extracts unique filter names from a list of filenames based on naming conventions."""
    return list(set(f.split('_')[position] for f in file_list if f.startswith(start)))

def log_memory_usage():
    """Prints the current resident set size (RSS) memory consumption of the script."""
    process = psutil.Process(os.getpid())
    print(f"Uso de memória: {process.memory_info().rss / 1024 ** 2:.2f} MB")