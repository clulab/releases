from math import exp

def textwrap(long_string, delimiter=" "):
    words = long_string.split(delimiter)
    spans = [words[x:x+10] for x in range(0,len(words),10)]
    span_strings = [delimiter.join(s) for s in spans]
    wrapped = "\n\t".join(span_strings)
    return wrapped


def load_model(filename, backoff):
    lines = open(filename).readlines()[1:]
    model = dict()
    for line in lines:
        fields = line.rstrip().split('\t')
        adj = fields[0]
        mu_coeff = float(fields[1])
        adj_int = float(fields[-1])

        # Add
        model[adj] = {"mu":mu_coeff, "intercept":adj_int}

        if not backoff:
            sigma_coeff = float(fields[2])
            model[adj]["sigma"] = sigma_coeff
    return model

def choose_model():

    use_stdev = getValidResponse("Include standard deviation?", ['y', 'n'])
    if use_stdev == 'y':
        backoff = False
    else:
        backoff = True

    model_type = getValidResponse("Do you want to use the full model or the high-frequency subset?", ["full", "hf"])

    models = {
        "full": {
            True: "backoff_model.txt",
            False: "full_model.txt"
        },
        "hf": {
            True: "highfreq_backoff_model.txt",
            False: "highfreq_model.txt"
        }
    }

    return load_model(models[model_type][backoff], backoff), backoff

def getValidResponse(prompt, options_list):
    is_valid = False
    response = None
    options_string = '/'.join([str(opt) for opt in options_list])
    options_string = textwrap(options_string, delimiter='/')
    while not is_valid:
        response = raw_input(prompt + " [" + options_string + "] ")
        if response in options_list:
            is_valid = True

    return response

def main():
    print "Gradable Adjectives Groundings Demo"

    predict_another = "y"

    model, backoff = choose_model()

    while predict_another == "y":

        user_adj = getValidResponse("Enter adjective: ", model.keys())
        user_mean = float(raw_input("Enter mean of item being modified: "))
        user_stdev = None
        if not backoff:
            user_stdev = float(raw_input("Enter stdev of item being modified: "))

        adjective_funct = model[user_adj]
        intercept = adjective_funct["intercept"]
        mu = adjective_funct["mu"]

        if not backoff:
            sigma = adjective_funct["sigma"]
            predicted = exp(intercept + (mu*user_mean) + (sigma*user_stdev)) * user_stdev
        else:
            predicted = exp(intercept + (mu * user_mean)) * user_mean
        print "predicted change (increase or decrease) from the mean: ", predicted

        predict_another = getValidResponse("Predict another?", ['y', 'n'])
        if predict_another == 'y':
            choose_another_model = getValidResponse("Choose another model?", ['y', 'n'])
            if choose_another_model == 'y':
                model, backoff = choose_model()



main()
