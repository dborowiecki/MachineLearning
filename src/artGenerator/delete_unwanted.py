import os
import csv
from os import listdir
from os.path import isfile,isdir, join
import load_config

'''
This script finds all not jpg files and remove them.
Also jpg with size bigger than 10MB 
'''

def not_jpg_in_dir(path):
	out = []
	pictures = [join(path,f) for f in listdir(path) if isfile(join(path,f))]

	for name in pictures:
		if not ".jpg" in name:
			out.add(name)


	return out;

def files_bigger_than(path, size_bytes):
	out = []
	pictures = [join(path,f) for f in listdir(path) if isfile(join(path,f))]

	for file in pictures:
		size_b = os.stat(file).st_size
		if size_b > size_bytes:
			out.append(file)


	return out;


config = load_config.get_config()
path_to_folder = config['path']#"/media/dmn/bbc6c909-5878-4a41-8590-51139f9491b2/artTainData/painter-by-numbers"

train_dir  	   = join(path_to_folder,'train')
validation_dir = join(path_to_folder,'validation')

genres = [f for f in listdir(train_dir) if isdir(join(train_dir, f))]

counter = 0
for genre in genres:
	trains = join(train_dir,genre)
	validations = join(validation_dir,genre)

	found  = not_jpg_in_dir(trains)
	found = found + not_jpg_in_dir(validations)
	for i in found:
		os.remove(i)

	found = files_bigger_than(trains	  , 10000000)
	found = found + files_bigger_than(validations, 10000000)
	counter =counter+len(found)
	print("Removing is suspeded, uncomment this code:")
	#Remove files bigger than 10MB from path and this with no .jpg extention
	#USE WITH CAUTION
	#for i in found:
	 	#os.remove(i)


