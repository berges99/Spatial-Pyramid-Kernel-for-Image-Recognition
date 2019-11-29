--------------------------------------------------------------------------
		  SPATIAL PYRAMID KERNEL (Machine Learning II) 
--------------------------------------------------------------------------

Here is a brew description of the attached files in the delivery folder:

- df128.csv, df64.csv: two .csv files containing the flattened image with reduced resolution 128x128 and 64x64

- results128.csv, results64.csv: two .csv files including all the results
for our experimentations shown in the paper

- hist1.jpg, hist2.jpg: two random images used for the visualization of
histogram intersection in 'Visualization.ipynb'

- Visualization.ipynb: IPython notebook used to generate all different
plottings in the paper
- Visualization.html: rendered version of the latter

- Load_and_Resize.ipynb: IPython notebook used to read all the raw data and
convert it into DataFrames and store it to .csv files.
- Load_and_Resize.html: rendered version of the latter
ACHTUNG!! Do not execute this notebook! We specifically included the .csv
of the processesed data to avoid executing the latter. Notice that the raw
image data is not included, and the execution of the latter may take over 30
minutes depending on the computer.

- run.py: Python script used for creating multiple subprocesses and executing
several SVMs in parallel in Google Cloud

###

- hist_svm.py: Main script. Automates the process of reading the data and
executes the SVM with the specified kernel parameters through the terminal.

Type: python3 hist_svm.py --help to get some useful hints on how the
parameters are passed.

It accepts 4 arguments:
	--data: csv file from which to read the data
	--L: L value used for partitioning
	--quantization: Level of quantization desired
	--train_frac: fraction of training samples to be taken
	--test_frac: fraction of test samples to be taken



