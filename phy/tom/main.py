import os, sys

if len(sys.argv) < 3:
    print "Please specify data directory and input CSV file."
    sys.exit()

workdir = sys.argv[1]
csv_file = sys.argv[2]

print "+--------------------------------------------------------+"
print "        Stage 1: Convert CSV Into .mat file"
print "+--------------------------------------------------------+"



