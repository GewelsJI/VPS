# Dataset Preparation

We first introduce a high-quality per-frame annotated VPS dataset, named SUN-SEG, which includes 158,690 frames elected from the famous [SUN dataset](http://amed8k.sundatabase.org). We extend the labels with diverse types, i.e., object mask, boundary, scribble, and polygon.

## Step-1: Request and Download

We could not distribute the video data in SUN-SEG due to the strict license issue of SUN. 

- Please follow the instruction on [SUN dataset](http://amed8k.sundatabase.org) to request SUN-dataset and download the dataset by yourself. Thank you for your understanding! After request and download the SUN dataset, you first unzip it via 
    
        cd ./data/
        unzip sundatabase_positive_part1.zip
        unzip sundatabase_positive_part2.zip
        mkdir SUN
        mv sundatabase_positive_part1/case* SUN
        mv sundatabase_positive_part2/case* SUN
        
    
    and move the positive images by cases to `./data/SUN/`. The file structure will be the same as below:

- As for SUN-SEG dataset, please download the dataset on [Google Drive]() and put it in `./data/` which is ready for re-organization.


        cd ./data/
        wget OUR_DATAURL --http-user=OUR_USERNAME  --http-passwd=OUR_PASSWORD 
        unzip SUN-SEG.zip

The file structure will be the same as below:

```
├──data
    ├──SUN
        ├──case1
            ├──IMAGE_NAME.jpg
            |...
        ├──case2
        |...
    ├──SUN-SEG
        ├──TrainDataset
            ├──GT
                ├──case1_1
                    ├──IMAGE_NAME.png
                    |...
            ├──Edge
                |...
            ├──Scribble
                |...
            ├──Polygon
                |...
            ├──Classification
                ├──classification.txt
            ├──Detection
                ├──bbox_annotation.json
        ├──TestEasyDataset
            ├──GT
            |...
        ├──TestHardDataset
            ├──GT
            |...
```

## Step-2: Re-organize the file structure

By running the script `download_and_reorganize.sh` 

    sh ./utils/download_and_reorganize.sh
        
, the original file structure in SUN-dataset will be re-organized to the same as SUN-SEG for better length balance. To the end, the folder `Frame` and `GT` will share the same file structure as shown below:

```
├──data
    ├──SUN-SEG
        ├──TrainDataset
            ├──Frame
                ├──case1_1
                    ├──IMAGE_NAME.jpg
                    |...
                ├──case1_3
                |...
            ├──GT
                ├──case1_1
                    ├──IMAGE_NAME.png
                    |...
                ├──case1_3
                |...
            ├──Edge
                |...
            ├──Scribble
                |...
            ├──Polygon
                |...
            ├──Classification
                ├──classification.txt
            ├──Detection
                ├──bbox_annotation.json
        ├──TestEasyDataset
            ├──Frame
                ├──case2_3
                |...
            ├──GT
                ├──case2_3
                |...
        ├──TestHardDataset
            ├──Frame
                ├──case1_2
                |...
            ├──GT
                ├──case1_2
                |...
```