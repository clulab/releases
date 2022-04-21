import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

"""
This script plots how many rules was the system able to generate from the clusters
It works on the output of @see TacredRuleGeneration scala app
"""

def learning_curves_plots(paths=['/data/nlp/corpora/odinsynth/data/TACRED/results/learning_curves/dynamic_p', '/data/nlp/corpora/odinsynth/data/TACRED/results/learning_curves/dynamic_r', '/data/nlp/corpora/odinsynth/data/TACRED/results/learning_curves/dynamic_f1']):
    paths = ['/home/rvacareanu/projects/temp/odinsynth/zz_temp.txt']
    with open(paths[0]) as fin:
        p = [float(x.strip().split(',')[0]) for x in fin.readlines()]
    with open(paths[0]) as fin:
        r = [float(x.strip().split(',')[1]) for x in fin.readlines()]
    with open(paths[0]) as fin:
        f1 = [float(x.strip().split(',')[2]) for x in fin.readlines()]

    l = len(p)
    # p = p[::5]
    # r = r[::5]
    # f1 = f1[::5]

    plt.plot(list(range(1, l + 1)), p)
    plt.xlabel("Number of rules used")
    plt.ylabel("Precision")
    # plt.title("Precision")
    plt.savefig("results/mar/all/p.png")
    plt.clf()

    plt.plot(list(range(1, l + 1)), r)
    plt.xlabel("Number of rules used")
    plt.ylabel("Recall")
    # plt.title("Recall")
    plt.savefig("results/mar/all/r.png")
    plt.clf()

    plt.plot(list(range(1, l + 1)), f1)
    plt.xlabel("Number of rules used")
    plt.ylabel("F1")
    # plt.title("F1")
    plt.savefig("results/mar/all/f1.png")
    plt.clf()


    plt.plot(list(range(1, l + 1)), p, label="Precision")
    plt.plot(list(range(1, l + 1)), r, label="Recall")
    plt.plot(list(range(1, l + 1)), f1, label="F1")
    plt.xlabel("Number of rules used")
    plt.ylabel("Performance")
    plt.legend()
    plt.savefig("results/mar/all/prf1.png")
    plt.clf()


def plots_from_cluster_logging():
    path = "/home/rvacareanu/projects/results/odinsynth_tacred5_current_static/"
    output_path = "results/mar/all/static"

    all_files = pd.read_csv(f"{path}/all_solutions.tsv", sep='\t').fillna('')

    all_files['solved'] = all_files['solved'].apply(lambda x: "Yes" if x == 1 else "No")

    solved = all_files[all_files['solved'] == 'Yes']
    unsolved = all_files[all_files['solved'] == 'No']

    # print(unsolved)
    # print(unsolved['spec_size'])
    # print(np.max(unsolved['spec_size'].tolist()))
    # exit()


    sns.histplot(solved, x='spec_size')
    plt.xlabel("Number of sentences (Solved)")
    plt.legend()
    plt.xlim(0,25)
    plt.savefig(f'{output_path}/solved_no_sentences.png')
    plt.clf()


    bins = np.max(unsolved['spec_size'].tolist())
    sns.histplot(unsolved, x='spec_size', bins=bins)
    plt.xlabel("Number of sentences (Unolved)")
    plt.xticks(range(bins))
    plt.legend()
    plt.savefig(f'{output_path}/unsolved_no_sentences.png')
    plt.clf()

    sns.histplot(all_files, x='solved')
    plt.xlabel("Solved?")
    plt.legend()
    plt.savefig(f'{output_path}/solved_vs_unsolved.png')
    plt.clf()


    sns.histplot(all_files, x='spec_size')# bins=np.unique(all_files['spec_size'].tolist()).shape[0])
    plt.xlabel("Number of sentences")
    plt.legend()
    plt.xlim(0,25)
    plt.savefig(f'{output_path}/no_sentences.png')
    plt.clf()

    sns.histplot(solved, x='num_steps')
    plt.xlabel("Number of steps when solved")
    plt.legend()
    plt.savefig(f'{output_path}/solved_in_steps.png')
    plt.clf()


    bins = np.unique((solved['max_highlight_length'] - solved['min_highlight_length']).tolist() + (unsolved['max_highlight_length'] - unsolved['min_highlight_length']).tolist())
    plt.hist(
        [(solved['max_highlight_length'] - solved['min_highlight_length']).tolist(), (unsolved['max_highlight_length'] - unsolved['min_highlight_length']).tolist()],
        color=['r', 'b'], 
        alpha=0.5, 
        label=['Solved', 'Unsolved'], 
        bins=np.arange(np.max(bins) + 1)-0.5,
        )
    plt.xlabel("Max - min highlight length")
    plt.legend()
    plt.savefig(f'{output_path}/max_min_highlight.png')
    plt.clf()


    bins = np.unique(solved['max_highlight_length'].tolist() + unsolved['max_highlight_length'].tolist())
    plt.hist(
        [solved['max_highlight_length'].tolist(), unsolved['max_highlight_length'].tolist()], 
        color=['r', 'b'], 
        alpha=0.5, 
        label=['Solved', 'Unsolved'], 
        bins=np.arange(np.max(bins) + 1)-0.5,
        )
    plt.xlabel("Max highlighted length")
    plt.xticks(range(np.max(bins) + 1))
    plt.legend()
    plt.savefig(f'{output_path}/max_highlight_length.png')
    plt.clf()



    print(all_files)

    print(bins)
    print(solved['max_highlight_length'].tolist() + unsolved['max_highlight_length'].tolist())


def print_stats():
    path = "/home/rvacareanu/projects/results/odinsynth_tacred5_current_static/"
    # path = "/home/rvacareanu/projects/results/odinsynth_tacred_generated_rules/dynamic_merged_all/"
    all_files = pd.read_csv(f"{path}/all_solutions.tsv", sep='\t').fillna('')

    all_files['solved'] = all_files['solved'].apply(lambda x: "Yes" if x == 1 else "No")

    solved = all_files[all_files['solved'] == 'Yes']
    unsolved = all_files[all_files['solved'] == 'No']

    num_steps = solved['num_steps']

    print(f"From a total of {all_files.shape[0]}, we solved {solved.shape[0]}")
    print(f"When we solve, we do so in: \n\tmedian={num_steps.median()}\n\tmean={num_steps.mean()}\n\tmax={num_steps.max()}\n\tmin={num_steps.min()}")
    print(unsolved['num_steps'].mean())
    print(unsolved['num_steps'].median())


    
    return 0


if __name__ == "__main__":
    # learning_curves_plots()
    print_stats()