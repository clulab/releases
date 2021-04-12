kb_path = "openbook.txt"

kb_data = list([])
with open(kb_path, 'r') as the_file:
    kb_data = [line.strip()[1:-1] for line in the_file.readlines()]

with open('sci_know_concat.txt', 'w') as f:
    for item in kb_data:
        f.write("%s\n" % item)