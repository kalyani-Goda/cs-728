import json


def tag_list_deducer(tag_sub_list):
	res = max(tag_sub_list,key=len)
	return res

with open('train.json') as f:     #change files as required
	data = json.load(f)

for i in range(len(data)):
	tags_list = data[i]['tags']
	temp_tags_list = []
	for item in tags_list:
		if isinstance(item,str):
			temp_tags_list.append(item)
		elif isinstance(item,list):
			temp_tags_list.append(tag_list_deducer(item))
	data[i]['tags'] = temp_tags_list


with open('train_data_preprocessed.json',"w") as final:
	json.dump(data,final,indent=2)

