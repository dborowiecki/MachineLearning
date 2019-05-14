import os
import csv
from os import listdir
from os.path import isfile,isdir, join
import load_config

'''
After dividing data by genre in sortDtasetByGenre.py, keras flow_from_direcory needs
separate folder for validation.
This script exctact images from train folder and transfer them to validation folder
'''

def move_painting_to_validation(genre, name):
	old = "./train/"+genre+"/"+name
	new_dir = "./validation/"+genre
	new = join(new_dir,name)


	if not os.path.isdir(new_dir):
		os.mkdir(new_dir)

	os.rename(old, new)


config = load_config.get_config()

percencate_of_validation = float(config['validation'])#0.25
path_to_folder = config['path']

train_dir = path_to_folder+'/train'
os.chdir(path_to_folder)

genres = [f for f in listdir(train_dir) if isdir(join(train_dir, f))]

for genre in genres:
	genre_path = join(train_dir,genre)
	paintings = [f for f in listdir(genre_path) if isfile(join(genre_path,f))]

	paintings_to_transfer = int(percencate_of_validation*len(paintings))

	for i in range(0,paintings_to_transfer):
		name = paintings[i]
		move_painting_to_validation(genre, name)
		print("Moved: "+genre+" "+name)
