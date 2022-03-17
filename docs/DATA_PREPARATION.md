# Dataset Preparation

We first introduce a high-quality per-frame annotated VPS dataset, named SUN-SEG, which includes 158,690 frames elected from the famous [SUN dataset](http://amed8k.sundatabase.org). We extend the labels with diverse types, i.e., object mask, boundary, scribble, and polygon.

## Step-1: Request

We could not distribute the video data in SUN-SEG due to the strict license issue of SUN. 

- Please follow the instruction on [SUN dataset](http://amed8k.sundatabase.org) to request SUN-dataset and download the dataset by yourself. Thank you for your understanding!
- xxxx google drive link

## Step-2: Download

After request and download the SUN dataset, you first unzip it via `aaa` and move the positive images by cases to `./data/SUN/`. The file structure will be the same as below:

```
├──data
    ├──SUN
        ├──case1
            ├──IMAGE_NAME.jpg
            |...
        ├──case2
        |...
```

## Step-3: Re-organize the file structure

By running the script `download_and_reorganize.sh`, the original file structure in SUN-dataset will be re-organized to the same as SUN-SEG for better length balance. To the end, the folder `Frame` and `GT` will share the same file structure as shown below:

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
            ├──[details other labels] johnson here
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

file path 

train.txt
easy.txt
hard.txt

classifcation 
train
    class1
        frame.png, ... [glob]
    class2
        img1.png, ..
    class3
        img1.png, ...