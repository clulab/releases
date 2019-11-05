"""Functions to load data from folders and augment it"""

import itertools
import logging
import os.path

from PIL import Image
import numpy as np
from torch.utils.data.sampler import Sampler
import torch
import random
from nltk.corpus import wordnet as wn

LOG = logging.getLogger('main')
NO_LABEL = -1 # 55 #### TODO: AJAY NOTE: To remove this .. only created to exclude NA  #

##################################################
#### RIEDEL DATASET LABELS
##################################################
# {'/broadcast/content/location': 0,
#  '/broadcast/producer/location': 1,
#  '/business/business_location/parent_company': 2,
#  '/business/company/advisors': 3,
#  '/business/company/founders': 4,
#  '/business/company/industry': 5,
#  '/business/company/locations': 6,
#  '/business/company/major_shareholders': 7,
#  '/business/company/place_founded': 8,
#  '/business/company_advisor/companies_advised': 9,
#  '/business/company_shareholder/major_shareholder_of': 10,
#  '/business/person/company': 11,
#  '/business/shopping_center/owner': 12,
#  '/business/shopping_center_owner/shopping_centers_owned': 13,
#  '/film/film/featured_film_locations': 14,
#  '/film/film_festival/location': 15,
#  '/film/film_location/featured_in_films': 16,
#  '/location/administrative_division/country': 17,
#  '/location/br_state/capital': 18,
#  '/location/cn_province/capital': 19,
#  '/location/country/administrative_divisions': 20,
#  '/location/country/capital': 21,
#  '/location/de_state/capital': 22,
#  '/location/fr_region/capital': 23,
#  '/location/in_state/administrative_capital': 24,
#  '/location/in_state/judicial_capital': 25,
#  '/location/in_state/legislative_capital': 26,
#  '/location/it_region/capital': 27,
#  '/location/jp_prefecture/capital': 28,
#  '/location/location/contains': 29,
#  '/location/mx_state/capital': 30,
#  '/location/neighborhood/neighborhood_of': 31,
#  '/location/province/capital': 32,
#  '/location/us_county/county_seat': 33,
#  '/location/us_state/capital': 34,
#  '/people/deceased_person/place_of_burial': 35,
#  '/people/deceased_person/place_of_death': 36,
#  '/people/ethnicity/geographic_distribution': 37,
#  '/people/ethnicity/included_in_group': 38,
#  '/people/ethnicity/includes_groups': 39,
#  '/people/ethnicity/people': 40,
#  '/people/family/country': 41,
#  '/people/family/members': 42,
#  '/people/person/children': 43,
#  '/people/person/ethnicity': 44,
#  '/people/person/nationality': 45,
#  '/people/person/place_lived': 46,
#  '/people/person/place_of_birth': 47,
#  '/people/person/profession': 48,
#  '/people/person/religion': 49,
#  '/people/place_of_interment/interred_here': 50,
#  '/people/profession/people_with_this_profession': 51,
#  '/sports/sports_team/location': 52,
#  '/sports/sports_team_location/teams': 53,
#  '/time/event/locations': 54,
#  'NA': 55}
##################################################

## GIDS dataset labels
##################################################
# {'NA': 4,
# '/people/person/place_of_birth': 3,
# '/people/person/education./education/education/institution': 2,
# '/people/deceased_person/place_of_death': 0,
# '/people/person/education./education/education/degree': 1}
##################################################

class RandomTranslateWithReflect:
    """Translate image randomly

    Translate vertically and horizontally by n pixels where
    n is integer drawn uniformly independently for each axis
    from [-max_translation, max_translation].

    Fill the uncovered blank area with reflect padding.
    """

    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, old_image):
        xtranslation, ytranslation = np.random.randint(-self.max_translation,
                                                       self.max_translation + 1,
                                                       size=2)
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size

        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = old_image.transpose(Image.ROTATE_180)

        new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))

        new_image.paste(old_image, (xpad, ypad))

        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop((xpad - xtranslation,
                                    ypad - ytranslation,
                                    xpad + xsize - xtranslation,
                                    ypad + ysize - ytranslation))

        return new_image


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

#randomly choose propotional labeled samples per class
def relabel_dataset_RE(dataset, args):
    unlabeled_idxs = []
    labeled_ids = []
    all_labels = list(enumerate(dataset.get_labels()))   # notice enumerate also keep the indexes
    random.shuffle(all_labels)  # randomizing the relabeling
    num_classes = dataset.get_num_classes()
    num_per_classes = dataset.get_num_per_classes()
    num_labels_per_class = []

    if args.labels.isdigit():   #integer --> number of labeled datapoints
        LOG.info("[relabel dataset] Choosing " + args.labels + " NUMBER OF EXAMPLES randomly as supervision")
        num_labels = int(args.labels)
        for i in range(num_classes):
            num_c = num_per_classes[i]
            num_labels_c = int(num_labels * num_c / len(all_labels))
            num_labels_per_class.append(num_labels_c)

    else:    #float number between 0 and 100 --> percentage
        LOG.info("[relabel dataset] Choosing " + args.labels + "% OF EXAMPLES randomly as supervision")
        percent_labels = float(args.labels)
        for i in range(num_classes):
            num_c = num_per_classes[i]
            num_labels_c = int(num_c * percent_labels / 100.0)
            num_labels_per_class.append(num_labels_c)

    for idx, l in all_labels:
        if num_labels_per_class[l] > 0:
            labeled_ids.append(idx)
            num_labels_per_class[l] -= 1
        else:
            unlabeled_idxs.append(idx)
            dataset.lbl[idx] = NO_LABEL

    LOG.info("[relabel dataset] Number of LABELED examples : " + str(len(labeled_ids)))
    LOG.info("[relabel dataset] Number of UNLABELED examples : " + str(len(unlabeled_idxs)))
    LOG.info("[relabel dataset] TOTAL : " + str(len(labeled_ids)+len(unlabeled_idxs)))
    return labeled_ids, unlabeled_idxs


#randomly choose propotional labeled samples per class
def relabel_dataset_nlp(dataset, args):
    unlabeled_idxs = []
    labeled_ids = []

    all_labels = list(enumerate(dataset.get_labels()))
    random.shuffle(all_labels) # randomizing the relabeling ...
    num_classes = dataset.get_num_classes()

    if args.labels.isdigit():
        # NOTE: if it contains whole numbers --> number of labeled datapoints
        LOG.info("[relabel dataset] Choosing " + args.labels + " NUMBER OF EXAMPLES randomly as supervision")
        num_labels = int(args.labels)
    else:
        # NOTE: if it contains a float (remember even xx.00) then it is a percentage ..
        #       give a float number between 0 and 100 .. indicating percentage
        LOG.info("[relabel dataset] Choosing " + args.labels + "% OF EXAMPLES randomly as supervision")
        percent_labels = float(args.labels)
        num_labels = int(percent_labels * len(all_labels) / 100.0)

    #to make sure that the labels are evenly distributed, from each class mark x number of labels as labeled,.
    num_labels_per_cat = int(num_labels / num_classes)

    labels_hist = {}
    for _, lbl in all_labels:
        if lbl in labels_hist:
            labels_hist[lbl] += 1
        else:
            labels_hist[lbl] = 1

    num_labels_per_cat_dict = {}
    for lbl, cnt in labels_hist.items():
        num_labels_per_cat_dict[lbl] = min(labels_hist[lbl], num_labels_per_cat)

    for idx, l in all_labels:
        if num_labels_per_cat_dict[l] > 0:
            labeled_ids.append(idx)

            #reduce the count of label/category which was stored in num_labels_per_cat_dict, by 1, every time you move a label as indexed.
            num_labels_per_cat_dict[l] -= 1
        else:
            #once you run out of all the count of labels that you had earmarked for labeling in a given category, mark the rest all as unlabeled.
            unlabeled_idxs.append(idx)
            dataset.lbl[idx] = NO_LABEL

    LOG.info("[relabel dataset] Number of LABELED examples : " + str(len(labeled_ids)))
    LOG.info("[relabel dataset] Number of UNLABELED examples : " + str(len(unlabeled_idxs)))
    LOG.info("[relabel dataset] TOTAL : " + str(len(labeled_ids)+len(unlabeled_idxs)))
    return labeled_ids, unlabeled_idxs


def get_all_label_indices(dataset):

    #this command returns all the labels and its corresponding indices eg:[198,2]
    all_labels = list(enumerate(dataset.get_labels()))

    #note that even though the labels are shuffled up, we are keeping track/returning only the shuffled indices. so it all works out fine.
    random.shuffle(all_labels)

    #get all the indices alone
    all_indices=[]
    for idx,_  in all_labels:
        all_indices.append(idx)
    return all_indices


def relabel_dataset(dataset, labels):
    unlabeled_idxs = []
    for idx in range(len(dataset.imgs)):
        path, _ = dataset.imgs[idx]
        filename = os.path.basename(path)
        if filename in labels:
            label_idx = dataset.class_to_idx[labels[filename]]
            dataset.imgs[idx] = path, label_idx
            del labels[filename]
        else:
            dataset.imgs[idx] = path, NO_LABEL
            unlabeled_idxs.append(idx)

    if len(labels) != 0:
        message = "List of unlabeled contains {} unknown files: {}, ..."
        some_missing = ', '.join(list(labels.keys())[:5])
        raise LookupError(message.format(len(labels), some_missing))

    labeled_idxs = sorted(set(range(len(dataset.imgs))) - set(unlabeled_idxs))

    return labeled_idxs, unlabeled_idxs


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size ## Note: floor division


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

class AddGaussianNoise:
    def __init__(self, stdev=1):
        self.stdev = stdev

    def __call__(self, x):
        noise = np.random.normal(scale=self.stdev, size=x.shape)
        return x + torch.Tensor(noise)


class TransformTwiceNEC:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp, entity_token):
        out1 = self.transform(inp, entity_token)
        out2 = self.transform(inp, entity_token)
        return out1, out2


class RandomPatternWordNoise:

    def __init__(self, number_words, replace, noise_type):
        self.number_words = number_words
        self.replace = replace
        self.noise_type = noise_type

    @staticmethod
    def replace_with_synonym(word_str):
        wordnet_synsets = wn.synsets(word_str)

        replacement = None
        for synset in wordnet_synsets:
            for lemma in synset.lemma_names():
                if lemma != word_str:
                    replacement = lemma
                    break
                if replacement is not None:
                    break

        print ("word: " + str(word_str) + " word replacement: " + str(replacement))
        return replacement

    def __call__(self, datums, entity_token):
        dropout_datums = list()
        for datum in datums:
            dropout_datum = list()
            to_replace = list(datum)

            if(entity_token in to_replace):
                to_replace.remove(entity_token)

            num_words_to_dropout = min(self.number_words, len(to_replace) - 1)
            if num_words_to_dropout > 0:
                to_replace = random.sample(to_replace, num_words_to_dropout)

            for w in datum:
                if w in to_replace:
                    if self.noise_type == 'drop':  # Dropout .. replace with NIL word
                        dropout_datum.append(self.replace)
                    else:  # Replace .. find a synonym of the word using wordnet
                        replaced_synonym = self.replace_with_synonym(w)
                        dropout_datum.append(replaced_synonym)
                else:
                    dropout_datum.append(w)
            dropout_datums.append(dropout_datum)

        return dropout_datums
