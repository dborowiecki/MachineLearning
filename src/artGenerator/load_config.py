

def get_config():
	config = {}
	with open("config") as config:
		lines = config.readlines()
		for line in lines:
			if line[0] is not "#":
				line.replace(": ", ":")
				variable, value = line.split(":")
				config[variable] = value

	return config;
