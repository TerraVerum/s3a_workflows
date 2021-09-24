from pathlib import Path
import os
import sys

def _init_fpic_folder():
    fpic = os.getenv('FPIC_FOLDER')
    if not fpic:
        take_next = False
        for arg in map(str.lower, sys.argv):
            if arg.strip('--') == 'fpic_folder':
                take_next = True
                if '=' in arg:
                    # Bound to this argument
                    fpic = arg.split('=')[-1]
                    break
            # Next argument is the folder
            if take_next:
                fpic = arg
                break
    if not os.path.exists(fpic):
        raise ValueError('Error: FPIC Path must be given and exist. Please run this script with'
                         '--fpic_folder <path-to-fpic-folder> or add FPIC_FOLDER to your environment variables.\n'
                         'This folder must contain `pcb_image` as a subdirectory')
    return fpic

FPIC_FOLDER = _init_fpic_folder()
