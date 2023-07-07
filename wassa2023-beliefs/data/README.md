#### README file for the Population Subjective Views dataset

Please direct correspondence regarding the dataset to Maria Alexeeva at alexeeva-at-arizona-dot-edu and Mihai Surdeanu at msurdeanu-at-arizona-dot-edu

Contents:

1. Description
2. File Name Structure
3. Annotation File Structure
4. Changelog


1. Description:

This is a training and testing dataset for evaluating a neural classifier that identifies sentences that contain population subjective views (beliefs and attitudes).

The dataset is described in the following publication (TODO: adjust when full citation is available):
```
Annotating and Training for Population Subjective Views. M. Alexeeva, C. Hyland, K. Alcock, A.A. Beal Cohen, H. Kanyamahanga, I. K. Anni, M. Surdeanu. In Proceedings of the 13th Workshop on Computational Approaches to Subjectivity, Sentiment & Social Media Analysis, 2023.
```

2. Directory Structure:

├── wassa2023_data_final_version
|   └── training_set
│        ├── known_triggers (6 files) # data annotated by MTurk and quality-controlled by the team
│        ├── unknown_triggers (10 files) # automatically sampled data containing no known triggers
│        ├── unknown_triggers_sample_used_for_experiments (1 file)
│   └── test_set
│        ├── known_triggers (1 file) # data annotated by the team in collaboration with domain experts
│        ├── unknown_triggers (1 file) # automatically sampled data containing no known triggers, annotated by the team; include a small set of sentences that are judged to be beliefs and include previously unknown belief triggers


3. File Structure

The data are stored in the `tsv` format. The following describes the header for the files in each directory

├── training set, known trigger files
|   └── uuid                          # data point unique ID
|   └── file                          # file name (not searchable online)
|   └── paragraph                     # paragraph-length context of the sentence annotated (automatically extracted)
|   └── sentence                      # sentence annotated
|   └── tokenized_sentence            # annotated sentence, tokenized
|   └── marked_sentence               # annotated sentence with the known belief trigger marked with special tokens, which highlight the trigger when used in html: <marked> (trigger start) and </marked> (trigger end)
|   └── approx_span                   # short span of text around the trigger
|   └── trigger                       # known belief trigger
|   └── marked_trigger                # known belief trigger marked with special tokens: <marked> (trigger start) and </marked> (trigger end)
|   └── submitted_annotations_count   # number of annotations submitted by MTurkers
|   └── accepted_count                # number of annotations not eliminated through initial filtering (see paper)
|   └── annotations                   # annotations submitted by MTurkers, <::>-separated
|   └── belief_ann_count              # number of sentences annotated as beliefs
|   └── majority                      # majority category (`Belief/attitude`, `Not belief/attitude`, or `tie`)
|   └── quality_controlled            # annotation from a team member (`b` for belief/attitude, `n` for not belief/attitude)
|   └── title                         # title of the paper
|   └── source                        # source of the paper (authors or organization)
|   └── year                          # paper creation date

├── training set, unknown trigger files
|   └── uuid                          # data point unique ID
|   └── file                          # file name (not searchable online)
|   └── sentence                      # sentence used as a negative data point (assumed to not contain a belief based on the absence of a known belief trigger)
|   └── tokenized_sentence            # negative data point sentence, tokenized
|   └── paragraph                     # paragraph-length context of the negative data point sentence (automatically extracted)
|   └── trimmed_sentence              # negative data point sentence with start- and end-of-sentence spaces removed
|   └── title                         # title of the paper
|   └── source                        # source of the paper (authors or organization)
|   └── year                          # paper creation date

├── training set, unknown_triggers_sample_used_for_experiments
|   └── sentence                      # sentence used as a negative data point (assumed to not contain a belief based on the absence of a known belief trigger)
|   └── paragraph                     # paragraph-length context of the negative data point sentence (automatically extracted)
|   └── quality_controlled            # annotation used in training; all data points assumed to be negative

├── test set, known trigger files
|   └── uuid                          # data point unique ID
|   └── file                          # file name (some file names are searchable online if the extension is changed from `txt` to `pdf`)
|   └── sentence                      # sentence annotated
|   └── tok_sents                     # annotated sentence, tokenized
|   └── trigger                       # known belief trigger
|   └── approx_span                   # short span of text around the trigger
|   └── quality_controlled            # annotation from a team member (`b` for belief/attitude, `n` for not belief/attitude)
|   └── paragraph                     # paragraph-length context of the sentence annotated (automatically extracted)
|   └── title                         # title of the paper
|   └── source                        # source of the paper (authors or organization)
|   └── year                          # paper creation date


├── test set, unknown trigger files
|   └── uuid                          # data point unique ID
|   └── file                          # file name (some file names are searchable online if the extension is changed from `txt` to `pdf`)
|   └── sentence                      # sentence annotated
|   └── tok_sent                      # annotated sentence, tokenized
|   └── trigger                       # known belief trigger, only available for sentences that contain beliefs
|   └── quality_controlled            # annotation from a team member (`b` for belief/attitude, `n` for not belief/attitude)
|   └── paragraph                     # paragraph-length context of the sentence annotated (automatically extracted)
|   └── title                         # title of the paper
|   └── source                        # source of the paper (authors or organization)
|   └── year                          # paper creation date


Please cite the following publication when using the dataset:

```
TODO
```

==========
Changelog:
==========

06/26/2023 - document created (alexeeva)
