#!/bin/bash


if [ $# -ne 1 ];
then
  echo "Usage: convert-tar <output-path>"
  exit 1
fi

read infile

# Get the file to tmp dir
pushd /tmp >/dev/null

tarfilename=`basename $infile`

beforedot=${tarfilename%.*}
afterhyphen=${beforedot#*-}

afterdot=${tarfilename#*.}

hadoop fs -copyToLocal $infile . 2>/dev/null

echo "copying file $infile to ."

echo "untarring $tarfilename"

tar -xf $tarfilename

echo "untared $tarfilename"

  pushd $afterhyphen >/dev/null
  for file in `ls *.JPEG`
  do
    #width=`identify -format "%w" $file`
    #height=`identify -format "%h" $file`
    #if [[ $(( $width * $height )) -ge 100000 ]];
    #then
	echo "Converting $file"
    convert -resize 256x256\! $file $file
    #fi
  done

  popd >/dev/null

tar cf $beforedot-scaled.tar $afterhyphen

rm -rf $afterhyphen
rm $tarfilename

hadoop fs -copyFromLocal $beforedot-scaled.tar $1/ 2>/dev/null

rm $beforedot-scaled.tar

popd >/dev/null
