"""Script handling user input for manual data labeling"""

__author__      = "Veronika Kalouskova"
__copyright__   = "Copyright 2025, FBMI CVUT"

import sys
import getopt

import data_handler
import gui


#   Print script usage in case of --help or incorrect usage
def usage():
    print('\ndata_labeler.py [-l segment_length] [-f sampling_rate] [-s starting_segment] input_file')
    print('      - input_file (str)       - input file name, must be a .csv file located in the /data directory')
    print('      - segment_length (int)   - length of a single segment in seconds, defaults to 10 seconds')
    print('      - sampling_rate (int)    - sampling frequency of input file, defaults to 500 frames per second')
    print('      - starting_segment (int) - segment number to display, defaults to fist segment')

    sys.exit(1)


#   Parse argument from the command line interface
def parse_argument(arg):
    if arg.isdigit():
        return int(arg)
    else:
        usage() 


#   Parse input from the command line interface
def parse_input(opts, args):
    # Default parameter values
    fs, seg_len, seg_num = 500, 10, 0

    # Parse optional arguments
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage()
        elif opt in ('-l', '--segment_length'):
            seg_len = parse_argument(arg)
        elif opt in ('-f', '--sampling_rate'):
            fs = parse_argument(arg)
        elif opt in ('-s', '--starting_sample'):
            seg_num = parse_argument(arg) - 1

    if seg_num < 0:
        seg_num = 0

    # Parse input file
    if not args:
        usage()
    else: 
        filename = args[0]
        dh = data_handler.DataHandler(filename, fs, seg_len)

    return dh, fs, seg_len, seg_num, filename


if __name__ == '__main__':
    try:
        long_options = ['help', 'segment_length=', 'sampling_rate=', 'starting_segment=']
        opts, args = getopt.getopt(sys.argv[1:], 'hl:f:s:', long_options)
    except getopt.GetoptError:
        usage()

    dh, fs, seg_len, seg_num, filename = parse_input(opts, args)
    
    gui.Application(dh, fs, seg_len, seg_num, filename)
