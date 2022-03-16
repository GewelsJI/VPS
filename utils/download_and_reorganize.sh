wget OUR_DATAURL --http-user=OUR_USERNAME  --http-passwd=OUR_PASSWORD

unzip SUN-SEG.zip

mv ./SUN-SEG ../data

python reorganize.py

rm -rf ../data/SUN