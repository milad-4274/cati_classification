#!/bin/sh 
dirname=.
find $dirname -type f -printf '%p/ %f\n' | sort -k2 | uniq -f1 --all-repeated=separate