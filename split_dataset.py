import splitfolders
input_file="C:/Users/patel/PythonProjects/potholes/dataset1"
output_file="C:/Users/patel/PythonProjects/potholes/split_dataset"
splitfolders.ratio(input_file,output=output_file,seed=38,ratio=(0.70,0.30),group_prefix=None)