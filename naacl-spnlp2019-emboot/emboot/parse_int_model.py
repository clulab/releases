# convert interpretable model into plottable format to match >:(

from collections import Counter

file = open("pools_output_Emboot.txt","r")
file_int = open("pools_output_interpretable.txt_interpretable_model.txt","r")

file2 = []
for line in file:
    line = line.strip()
    line = line.split("\t")
    #print(line)
    if line[0] != "Epoch":
        file2.append(line[1:])
    else:
        continue
#for line in file2:
#    print(line)
#print(file2)

entities = []
for line in file2:
    for entity in line:
        entities.append(entity)
#print(entities)
print(len(entities))

file_int_2 = []
for line in file_int:
    line = line.strip()
    line = line.split("\t")
    if line[0] == "20":
        file_int_2.append(line[1:])
    else:
        continue

entities_int = []
for line in file_int_2:
    entities_int.append(line[0])

#print(entities_int)
print(len(entities_int))
ents = set(entities_int)

print(len(ents))


extra_entities = []
overlap_entities = []
for entity in entities_int:
    if entity in entities:
        overlap_entities.append(entity)
    else:
        extra_entities.append(entity)
print("Overlapping entities: {}".format(len(overlap_entities)))
print("Extra entities in interpretable model: {}".format(len(extra_entities)))
#print(extra_entities)
#[print(item) for item in sorted(overlap_entities)]
#[print(item) for item in sorted(extra_entities)]

for entity in sorted(entities):
    if entity not in entities_int:
        print(entity)