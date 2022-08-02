#Various Utilities for handling the files
import glob
import os
 
def CombineFiles(directory, base, new_ending='_allvars_'):
    """
    Generates a file with all the combined formulas from the individual files from the base.
 
    Parameter:
    directory: directory files are in ("" for same directory as kernel is running)
    base: base name of files
    new_ending: the name replacing the ending of the file
 
    Returns:
    None
    """
    if os.path.exists(directory+base+new_ending+".txt"):
        os.remove(directory+base+new_ending+".txt")
    globs = glob.glob(directory+base+"*.txt")
    with open(directory+base+new_ending+".txt", 'w') as nf:
        for fname in globs:
            with open(fname, 'r') as f:
                nf.write(f.read())
            os.remove(fname)