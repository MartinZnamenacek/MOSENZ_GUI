
### Data description
* timestamp_str        - timestamp
* signals[0] as i32    - ECG
* signals[1] as i32    - O-N
* signals[2] as i32    - Th 1 RAW
* signals[3] as i32    - Th 2 RAW
* signals[4]           - Th 1 filtered
* signals[5]           - Th 2 filtered

### /src/data_labeler

The script and graphical user interface for manual data labelling are located in this directory. The GUI can be run from the */data_labeler* directory using the command __python3 data_labeler.py input_file__.

data_labeler.py [-l segment_length] [-f sampling_rate] [-s starting_segment] input_file
* input_file (str)       - input file name, must be a .csv file located in the */data* directory
* segment_length (int)   - length of a single segment in seconds, defaults to 10 seconds
* sampling_rate (int)    - sampling frequency of input file, defaults to 500 frames per second
* starting_segment (int) - segment number to display, defaults to fist segment

__EXAMPLE: *python3 data_labeler.py -l 10 -f 500 -s 1 01DEDOKA.csv*__

__NOTE: *Instead of using the GUI buttons for switching between segments and labelling one can opt to use keyboard shortcuts:
 * Left key and right key for switching between segments
 * Space key for toggling the confidence toggle button 
 * 1 key for toggling the reference toggle button
 * 2 key for toggling the first signal toggle button
 * 3 key for toggling the second signal toggle button
 * Enter key for leaving the application*__


