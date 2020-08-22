#!/bin/bash

ROOT=/
HTTP="/"
OUTPUT="index.html" 


echo "<!DOCTYPE html><html lang="en"><head><title>emgfit documentation</title><link href="https://fonts.googleapis.com/css?family=Nunito" rel="stylesheet"><style type="text/css">body {" > $OUTPUT
echo "  font-family: 'Nunito', sans-serif;" >> $OUTPUT
echo "  color: rgba(0, 0, 0, 0.87);" >> $OUTPUT
echo "  background-color: #FAFAFA;" >> $OUTPUT
echo "}" >> $OUTPUT
echo "h1 {" >> $OUTPUT
echo "  font-size: 45px;" >> $OUTPUT
echo "  font-weight: 500;" >> $OUTPUT
echo "  text-align: center;" >> $OUTPUT
echo "}" >> $OUTPUT
echo "li {" >> $OUTPUT
echo "  list-style-type: none;" >> $OUTPUT
echo "  text-align: center;" >> $OUTPUT
echo "}</style></head><body><h1>emgfit documentation</h1>" >> $OUTPUT

i=0
echo "<UL>" >> $OUTPUT
for filepath in `find "$ROOT" -maxdepth 1 -mindepth 1 -type d| sort`; do
  path=`basename "$filepath"`
  echo "  <LI><a href=\"/$path\">$path</a></LI>" >> $OUTPUT
done
echo "</UL>" >> $OUTPUT
echo "</ul></body></html>" >> $OUTPUT
