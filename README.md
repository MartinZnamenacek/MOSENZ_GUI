
### Data description
* timestamp_str        - timestamp
* signals[0] as i32    - ECG
* signals[1] as i32    - O-N
* signals[2] as i32    - Th 1 RAW
* signals[3] as i32    - Th 2 RAW
* signals[4]           - Th 1 filtered
* signals[5]           - Th 2 filtered

### /src/data_labeler

The script and graphical user interface for manual data labelling are located in this directory. The GUI can be run from the */data_labeler* directory using the command:\
`python data_labeler.py [-l segment_length] [-f sampling_rate] [-s starting_segment] input_file`
* segment_length (int)   - length of a single segment in seconds, defaults to 10 seconds
* sampling_rate (int)    - sampling frequency of input file, defaults to 500 frames per second
* starting_segment (int) - segment number to display, defaults to fist segment
* input_file (str)       - input file name, must be a .csv file located in the */data* directory

EXAMPLE: `python data_labeler.py 01DEDOKA.csv`\
EXAMPLE: `python data_labeler.py -l 10 -f 500 -s 1 01DEDOKA.csv`

Instead of using the GUI buttons for switching between segments and labelling, one can opt to use keyboard shortcuts:
 * `WASD` keys to switch between segments (`W/D` forwards, `A/S` backwards)
 * `C` key to toggle confidence ON & OFF
 * `1` key to toggle reference visibility ON & OFF
 * `2` key to toggle Th1 signal visibility ON & OFF
 * `3` key to toggle Th2 signal visibility ON & OFF
 * `Enter` key to leave the application
