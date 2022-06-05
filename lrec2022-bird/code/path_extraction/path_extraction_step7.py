from ast import literal_eval
import math

def main():

    files_dir = "../data/"
    pathids_to_paths_file = files_dir + "pathids_to_paths.tsv"
    f = open(pathids_to_paths_file, mode="r", encoding="utf_8")
    pathids_to_paths = {}
    lines = f.readlines()
    for line in lines:
        fields = line.strip().split("\t")
        path_id = literal_eval(fields[0])
        path = literal_eval(fields[1])
        pathids_to_paths[path_id] = get_path_textual_string(path)
    f.close()

    paths_features_file = files_dir + "paths_features.tsv"
    f = open(paths_features_file, mode="r", encoding="utf_8")
    paths_features = {}
    lines = f.readlines()
    for line in lines:
        fields = line.strip().split("\t")
        path_id = literal_eval(fields[0])
        slotX = literal_eval(fields[1])
        slotY = literal_eval(fields[2])
        paths_features[path_id] = (slotX,slotY)
    f.close()

    
    xfeatures_paths_file = files_dir + "xfeatures_paths.tsv"
    f = open(xfeatures_paths_file, mode="r", encoding="utf_8")
    xfeatures_paths = {}
    lines = f.readlines()
    for line in lines:
        fields = line.strip().split("\t")
        word = literal_eval(fields[0])
        path_ids = literal_eval(fields[1])
        xfeatures_paths[word] = path_ids
    f.close()
    
    yfeatures_paths_file = files_dir + "yfeatures_paths.tsv"
    f = open(yfeatures_paths_file, mode="r", encoding="utf_8")
    yfeatures_paths = {}
    lines = f.readlines()
    for line in lines:
        fields = line.strip().split("\t")
        word = literal_eval(fields[0])
        path_ids = literal_eval(fields[1])
        yfeatures_paths[word] = path_ids
    f.close()
    
    
    print("Creating paths slots frequency dictionary...")
    paths_slotfreq = {}
    i = 0
    progressMileStone = 0.05
    total = len(paths_features)
    for path_id,(slotX,slotY) in paths_features.items():
        if ((i/total) > progressMileStone):
            print(str(round(progressMileStone * 100)) + "% ", end='', flush=True)
            progressMileStone += 0.05
    
        slotx_freq = 0
        for f in slotX.values():
            slotx_freq += f
            
        sloty_freq = 0
        for f in slotY.values():
            sloty_freq += f
            
        paths_slotfreq[path_id] = (slotx_freq , sloty_freq)
        
        i += 1
    print("100%\n", flush=True)
    
    print("Writing paths slots frequency to disk...")
    f = open(files_dir + "paths_slotfreq.tsv", "w")
    i = 0
    progressMileStone = 0.05
    delim = "\t"
    for path_id,(slotx_freq,sloty_freq) in paths_slotfreq.items():
        if ((i/total) > progressMileStone):
            print(str(round(progressMileStone * 100)) + "% ", end='', flush=True)
            progressMileStone += 0.05
        
        f.write(str(path_id) +
                delim +
                "(" + str(slotx_freq) + "," + str(sloty_freq) + ")" +
                "\n")
        
        i += 1
    print("100%\n", flush=True)
    f.close()


    print("\nCreating total features (words) frequency for slotX and slotY...")
    xfeatures_totalfreqs = {}
    i = 0
    progressMileStone = 0.05
    total = len(xfeatures_paths)
    for w,pids in xfeatures_paths.items():
        if ((i/total) > progressMileStone):
            print(str(round(progressMileStone * 100)) + "% ", end='', flush=True)
            progressMileStone += 0.05

        xfeature_totalfreq = 0
        for pid in pids:
            xfeature_totalfreq += paths_features[pid][0][w]
            
        xfeatures_totalfreqs[w] = xfeature_totalfreq
        
        i += 1
    print("100%", flush=True)        
    
    yfeatures_totalfreqs = {}
    i = 0
    progressMileStone = 0.05
    total = len(yfeatures_paths)
    for w,pids in yfeatures_paths.items():
        if ((i/total) > progressMileStone):
            print(str(round(progressMileStone * 100)) + "% ", end='', flush=True)
            progressMileStone += 0.05

        yfeature_totalfreq = 0
        for pid in pids:
            yfeature_totalfreq += paths_features[pid][1][w]
        
        yfeatures_totalfreqs[w] = yfeature_totalfreq
        
        i += 1
    print("100%\n", flush=True)
    
    print("Writing total features frequency for slotX and slotY to disk...")
    
    f = open(files_dir + "xfeatures_totalfreqs.tsv", mode="w", encoding="utf_8")
    delim = "\t"
    for w,freq in xfeatures_totalfreqs.items():
        f.write('"' + process_string(w) + '"' +
                delim +
                str(freq) +
                "\n")
    f.close()
    
    f = open(files_dir + "yfeatures_totalfreqs.tsv", mode="w", encoding="utf_8")
    for w,freq in yfeatures_totalfreqs.items():
        f.write('"' + process_string(w) + '"' +
                delim +
                str(freq) +
                "\n")
    f.close()

    
def get_path_textual_string(path):
    
    ret_val = ""
    last_printed_element = ""
    first_element_to_print = ""
    
    for t in path:
        if (t[3] == ">"):
            first_element_to_print = t[0]
            if (first_element_to_print == last_printed_element):
                first_element_to_print = ""
                       
            ret_val += first_element_to_print + "->" + t[2] + "->" + t[1]
            last_printed_element = t[1]
        else:
            first_element_to_print = t[1]
            if (first_element_to_print == last_printed_element):
                first_element_to_print = ""
                    
            ret_val += first_element_to_print + "<-" + t[2] + "<-" + t[0]
            last_printed_element = t[0]
            
    return ret_val[1:-1]


def process_string(s):
    ret_val = s.replace("\\" , "\\\\")
    ret_val = ret_val.replace("\"" , "\\\"")
    return ret_val
   
    
if __name__ == "__main__":
    main()
