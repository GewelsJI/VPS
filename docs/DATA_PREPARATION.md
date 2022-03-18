# Dataset Preparation

# Contents
- [Step-1: Request and Download](#step-1--request-and-download)
- [Step-2: Unzip SUN dataset](#step-2--unzip-sun-dataset)
- [Step-3: Re-organize the file structure](#step-3--re-organize-the-file-structure)

We first introduce a high-quality per-frame annotated VPS dataset, named SUN-SEG, which includes 158,690 frames elected from the famous [SUN dataset](http://amed8k.sundatabase.org). We extend the labels with diverse types, i.e., object mask, boundary, scribble, and polygon.

# Step-1: Request and Download

> **Note:** The origin colonoscopy video frames in our SUN-SEG dataset are selected from [SUN dataset](http://amed8k.sundatabase.org), while we could not distribute the video data due to the strict license. 

So first, you guys need to request the origin colonoscopy video frame from them. In this step, you should download the polyp samples of 100 cases and non-polyp samples of 13 cases from the links provided by SUN dataset. 

- **Request for video frames from SUN:** Please follow the instruction on [SUN dataset](http://amed8k.sundatabase.org) to request SUN-dataset and download the dataset by yourself. Thank you for your understanding!

And then, you can feel free to download the complete annotation provided by our SUN-SEG.

- **Request for annotations from SUN-SEG:** Our re-organized annotations could be downloaded at [google drive]().

# Step-2: Unzip SUN dataset

As for video frames in SUN dataset, these are two groups of samples, which are divided into multiple compressed file format as zip files. To decompress each zip file downloaded, please input the password provided by origin authors of SUN dataset (i.e., the same as the password that you used for login).

- **Unzip positive cases in SUN**
    - create directory: `mkdir ./data/SUN-Positive/`
    - unzip positive cases: `unzip -P sun_password -d ./SUN-Positive sundatabase_positive_part\*`, which will take up 11.5 + 9.7 GB of storage space. Please ensure your server has enough space to storage them, otherwise it will fail. Please replace the `sun_password` what you get from SUN's authors.
    - check if correct: `find ./SUN-Positive -type f -name "*.jgp" | wc -l`, which should output 49,136 in your terminal.
- **Unzip negative cases in SUN (Optional)**
    - create directory: `mkdir ./data/SUN-Negative/`
    - unzip negative cases: `unzip -P sun_password -d ./SUN-Negative sundatabase_positive_part\*`, which will take up 11.6 + 10.7 + 11.5 + 10.5 GB of storage space. (This data partition is optional if we have no requirments to use them.)
    - check if correct: `find ./SUN-Negative -type f -name "*.jgp" | wc -l`, which should output 109,554 in your terminal.

After prepare all the files,  your file structure will be the same as below:

```
├──data
    ├──SUN-Positive
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

As for the annotations from our SUN-SGE, you are happy to execute:

- Unzip it via `unzip SUN-SEG-Annotation.zip`
- Put it at path `./data/SUN-SEG-Annotation/` 

Then, your file structure will be the same as below:

```
昱程：添加你的文件结构在这里
```


# Step-3: Re-organize the file structure

By running `python ./utils/reorganize.py`, the original file structure in SUN-dataset will be re-organized to the same as SUN-SEG for better length balance. Finally, the folder `Frame` and `GT` will share the same file structure as shown below:

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