# UndiagnosedDiseaseClassifier
Detect users who post about undiagnosed diseases and refer them to the Undiagnosed Disease Network (UDN).

This is a project for Advanced Social Computing under Hadi Amiri.

Project members: Qilei Chen, Zijun He, David Donahue

## Goal of the Project 

Combine multiple sources of information such as Reddit, Twitter, and UDN profiles to train a model
for classifying posts/comments about undiagnosed diseases.

For further details on the project, please have a look at the final paper COMP5800_Final_Report.pdf

## Details of Components

dump_filter.py - if dumps are downloaded from https://files.pushshift.io/reddit/submissions/ this script filters them for various keywords to extract training data

filter_reddit_data.py - uses Reddit API to manually select Reddit submissions which contain an undiagnosed disease as training data

get_random_submissions.py - get random submissions as negative samples to evaluate model accuracy

train_undiagnosed_classifier.py - use data gathered from previous sources to train and evaluate an undiagnosed disease classifier
