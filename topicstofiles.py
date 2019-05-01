#!/usr/bin/python

# topicstofiles.py: Prints the words that are most prominent in a set of
# topics and saves them into a text file, for all topics in a directory.
#
# This code suite is largely adapted from the online VB (aka stochastic
# variational Bayes) code of
# Matthew D. Hoffman, Copyright (C) 2010
# found here: http://www.cs.princeton.edu/~blei/downloads/onlineldavb.tar
# and also of 
# Chong Wang, Copyright (C) 2011
# found here: http://www.cs.cmu.edu/~chongw/software/onlinehdp.tar.gz

import sys, numpy, scipy.io, warnings, glob

# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
dirname = sys.argv[1]  # the directory we want to process
filenames = glob.glob(dirname + "/*.npy")  # find all files ending with .npy
for filename in filenames:
    # Load vocab
    if ("nature" in filename):
        vocab = str.split(file("nature_vocab.dat").read())
    elif ("wiki" in filename):
        vocab = str.split(file("wiki_vocab.dat").read())
    else:
        print ""
        print "Cannot determine vocab to use. Skipping " + filename + "..."
        continue

    # Prepare file to write on
    txtfilename = filename[0:-4] + ".txt"  # replace .npy with .txt
    f = open(txtfilename, 'w')
    f.write(filename + "\n\n")

    # Iterate over topics and print the top words
    numtopwords = 15
    lam = numpy.load(filename)
    for k in range(0, len(lam)):
        lambda_k = list(lam[k, :])
        lambda_k = lambda_k / sum(lambda_k)
        temp = zip(lambda_k, range(0, len(lambda_k)))
        temp = sorted(temp, key = lambda x: x[0], reverse=True)
        f.write("#" + str(k+1) + " " + " ".join([vocab[t[1]] + "(" + ("%g" % t[0]) + ")" for t in temp[0:numtopwords]]) + "\n\n")

    f.close()
    print "Converted " + filename
