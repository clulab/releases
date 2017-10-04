from math import exp

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



def main():
    print "Gradable Adjectives Groundings Demo"
    backoff = False
    valid_resp = False
    while not valid_resp:
        use_stdev = raw_input("Include standard deviation? [y/n] ")
        if use_stdev == 'y':
            backoff = False
            valid_resp = True
        elif use_stdev == 'n':
            backoff = True
            valid_resp = True
        else:
            print "Invalid reponse."
    valid_resp = False
    model_type = None
    while not valid_resp:
        model_type = raw_input("Do you want to use the full model or the high-frequency subset? [full/hf] ")
        if model_type == 'full' or model_type == "hf":
            valid_resp = True
        else:
            print "Invalid reponse."

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

    model = load_model(models[model_type][backoff], backoff)

    user_adj = raw_input("Enter adjective: ")
    if user_adj not in model:
        print "Sorry, that adjective ({0}) is not in our model.  Please choose from:\n{1}".format(user_adj, model.keys())
    user_mean = float(raw_input("Enter mean of item being modified: "))
    user_stdev = None
    if not backoff:
        user_stdev = float(raw_input("Enter stdev of item being modified: "))

    adjective_funct = model[user_adj]
    intercept = adjective_funct["intercept"]
    # print "intercept type: ", type(intercept), intercept
    mu = adjective_funct["mu"]
    # print "mu type: ", type(mu), mu

    if not backoff:
        sigma = adjective_funct["sigma"]
        # print "sigma type: ", type(sigma), sigma
        predicted = exp(intercept + (mu*user_mean) + (sigma*user_stdev)) * user_stdev
    else:
        predicted = exp(intercept + (mu * user_mean)) * user_mean
    print "predicted change (increase or decrease) from the mean: ", predicted




main()
