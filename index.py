a_file = open("sample.json", "r")
a_json = json.load(a_file)
pretty_json = json.dumps(a_json, indent=4)
a_file.close()