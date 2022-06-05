import os
import sys

def main():
    files_dir = "../data/raws/"
    os.chdir(files_dir)
    
    for i in range(1,101):
        process_corpus_file("corpus_part" + str(i).zfill(3))
        if (i % 5 == 0):
            print(str(i) + "% ", end='', flush=True)


def process_corpus_file(corpus_filename):
    f = open(corpus_filename, encoding="utf-8", mode="r")
    lines = f.readlines()
    f.close()
    
    for i in range(len(lines)):
        
        lines[i] = lines[i].strip()
        lines[i] += "\n"
        
        if (len(lines[i]) > 1):
            if (lines[i][-2] not in [".","!",";","?"]):
                lines[i] = lines[i][0:-1] + "." + lines[i][-1]
                
    os.system("rm " + corpus_filename)
    f = open(corpus_filename, encoding="utf-8", mode="w")
    f.writelines(lines)
    f.close()


if __name__ == "__main__":
    main()
