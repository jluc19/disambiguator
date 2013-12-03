import langid

if __name__ == '__main__':
	f = open('diabetes.txt', 'r')
	file_content = f.readlines()
	w = open('diabetes_parsed.txt', 'w')

	non_dup = list(set(file_content))
	for line in non_dup:
		tup = langid.classify(line)
		if "en" in tup:
			if not line.startswith("RT"):
				if not "http" in line:
					w.write(line)

	f.close()
	w.close()