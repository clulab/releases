import pickle

def print_example(instance):
    print("-"*40)
    print("input:")
    for input_text in instance["input"].split(". "):
        print(input_text)
    print("\n")
    print("output:", instance["output"])

    input("-"*20)

    return 0


def load_data_and_check():

    with open("saved_data/20210205/harvested_train_examples_epoch_0.pickle" , "rb") as handle:
        instances = pickle.load(handle)

    for module in ["f", "r"]:
        print("="*80)
        print("checking module "+module)
        print("total number of examples in the module:", len(instances[module]))
        print("\n")

        for instance in instances[module]:
            print_example(instance)

    return 0

def load_data_and_check_statistics():

    #correct_instance_token_dict = {}
    #incorrect_instance_token_dict = {}

    stop_words = {'episodic': 100, 'buffer:': 100, 'there': 50, 'are': 50, '2': 39, 'fact': 324, 'buffers': 100,
                  'and': 50, '1': 103, 'rule': 50, '': 391, 'i': 50, 'want': 50, 'to': 50, 'judge': 50, 'whether': 50,
                  'buffer': 50, 'does': 57, 'not': 122, 'contradict': 46, 'is': 233, '1:': 43,
                  '2:': 40, '3:': 32, '4:': 24, '5:': 20,
                  'operator:': 50, 'RUN': 50, '</s>': 100, 'true,': 13, 'NAF': 13, '6:': 7,
                  '7:': 7, '8:': 6, '9:': 6, '10:': 6, 'false,': 37, 'this': 33,
                  'contradicted': 33, 'by': 33, '7': 5,
                  'can': 4, 'prove': 4,
                  'CWA': 4, 'the': 15, '3': 29,
                  '5': 1, '10': 1, '8': 1}

    with open("saved_data/20210205/harvested_train_examples_epoch_0.pickle" , "rb") as handle:
        instances = pickle.load(handle)

    for module in ["f", "r"]:
        print("="*80)
        print("checking module "+module)
        print("total number of examples in the module:", len(instances[module]))
        print("\n")

        instance_token_dict = {}

        for instance in instances[module]:
            input_tokens = instance["input"].replace(".", " ").replace("\"", " ").split(" ")
            for input_token in input_tokens:
                if input_token not in stop_words:
                    if input_token not in instance_token_dict:
                        instance_token_dict[input_token] = 1
                    else:
                        instance_token_dict[input_token] +=1

            output_tokens = instance["output"].replace("."," ").replace("\"", " ").split(" ")
            for output_token in output_tokens:
                if output_token not in stop_words:
                    if output_token not in instance_token_dict:
                        instance_token_dict[output_token] = 1
                    else:
                        instance_token_dict[output_token] +=1


        print("dict:", instance_token_dict)
        input("-------")


#load_data_and_check()
load_data_and_check_statistics()