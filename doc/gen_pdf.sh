#!/usr/bin/env bash

# silence
if [[ "$1" == "-s" ]]
then
out=/dev/null
else
out=/dev/stdout
fi

base=`pwd`
doc=`dirname $0`

cd $doc
mkdir -p ./target
find -name "*.tex" -print0 | xargs -0 -I texfile  latex -output-directory=./target -output-format=pdf texfile > $out

cd $base