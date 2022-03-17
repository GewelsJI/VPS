# Dataset Preparation

We first introduce a high-quality per-frame annotated VPS dataset, named SUN-SEG, which includes 158,690 frames elected from the famous [SUN dataset](http://amed8k.sundatabase.org). We extend the labels with diverse types, i.e., object mask, boundary, scribble, and polygon.

## Request 

We do not distribute the dataset because of the license issue. Please follow the instruction on [SUN dataset](http://amed8k.sundatabase.org) to request SUN-dataset and download the dataset by yourself.

## Download

After request SUN dataset, you first move the positive images by cases to `./data/SUN/`. The file structure will be the same as below:

```
├──data
    ├──SUN
        ├──case1
            ├──IMAGE_NAME.jpg
            |...
        ├──case2
        |...
```

## Re-organize the file structure

By running the script `download_and_reorganize.sh`, the SUN-SEG dataset will be downloaded and the original file structure in SUN-dataset will be re-organized to the same as SUN-SEG for better length balance. In the end, the folder `Frame` and `GT` will share the same file structure as shown below:

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
