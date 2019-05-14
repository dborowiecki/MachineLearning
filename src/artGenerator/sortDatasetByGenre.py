import os
import csv
from os import listdir
from os.path import isfile, join
import load_config
#moving to subfolders
def moveFileToSubfolder(filename, foldername):
	old = "./train/"+filename
	new_dir = "./train/"+foldername
	new = new_dir+"/"+filename

	if not os.path.isdir('./train/'+foldername):
		os.mkdir(new_dir)

	os.rename(old, new)


config = load_config.get_config()
#because files are stored on external disc
path_to_folder = config['path']
os.chdir(path_to_folder)
onlyfiles = [f for f in listdir(path_to_folder) if isfile(join(path_to_folder, f))]

#check csv files and move to genre subfolders
with open('train_info.csv', 'r') as painting_data:
	reader = csv.DictReader(painting_data, delimiter=',')
	i = 0
	for row in reader:
		#print(row['filename']+' '+row['genre'])
		moveFileToSubfolder(row['filename'], row['genre'])