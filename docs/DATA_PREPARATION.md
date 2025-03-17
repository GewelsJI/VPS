# Dataset Preparation

We introduce a high-quality per-frame annotated VPS dataset, named SUN-SEG, which includes 158,690 frames elected from the famous [SUN dataset](http://amed8k.sundatabase.org). Then, we extend the labels with diverse types, i.e., object mask, boundary, scribble, and polygon. If you want to get access to our whole dataset, you should follow the next three steps.

# Contents
- [Step-1: Request and Download](#step-1--request-and-download)
- [Step-2: Unzip SUN dataset](#step-2--unzip-sun-dataset)
- [Step-3: Re-organize the file structure](#step-3--re-organize-the-file-structure)


# Step-1: Request and Download

> **Note:** The origin colonoscopy video frames in our SUN-SEG dataset are selected from [SUN dataset](http://amed8k.sundatabase.org), while we could not distribute the video data due to the strict license. 

So first, you guys need to request the original colonoscopy video frame from them. In this step, you should download the polyp samples of 100 cases and non-polyp samples of 13 cases from the links provided by the SUN dataset. 

- **Request for video frames from SUN:** Please follow the instruction on [SUN dataset](http://amed8k.sundatabase.org) to request SUN-dataset and download the dataset by yourself. **Please use your educational email to apply for it and claim it without any commercial purpose.** Thank you for your understanding!

And then, you can feel free to download the complete annotation provided by our SUN-SEG.

- **Request for annotations from SUN-SEG:** Our re-organized annotations can be downloaded at the download link:
    - (July 17, 2024) Updated link: [annotation.v2 - google drive](https://drive.google.com/file/d/1ytmhpg0YaW0XZBAEfSFkMhaI6jnGLSg3/view?usp=sharing)

# Step-2: Unzip SUN dataset

As for video frames in the SUN dataset, these are two groups of samples, which are divided into multiple compressed file formats as zip files. To decompress each zip file downloaded, please input the password provided by the origin authors of the SUN dataset (i.e., the same as the password that you used for login).

- **Unzip positive cases in SUN**
    - create directory: `mkdir ./data/SUN-Positive/`
    - unzip positive cases: `unzip -P sun_password -d ./SUN-Positive sundatabase_positive_part\*`, which will take up 11.5 + 9.7 GB of storage space. Please ensure your server has enough space to store them, otherwise, it will fail. Please replace the `sun_password` that you get from SUN's authors.
    - check if correct: `find ./SUN-Positive -type f -name "*.jpg" | wc -l`, which should output 49,136 in your terminal.
- **Unzip negative cases in SUN (Optional)**
    - create directory: `mkdir ./data/SUN-Negative/`
    - unzip negative cases: `unzip -P sun_password -d ./SUN-Negative sundatabase_positive_part\*`, which will take up 11.6 + 10.7 + 11.5 + 10.5 GB of storage space. (This data partition is optional if we have no requirements to use them.)
    - check if correct: `find ./SUN-Negative -type f -name "*.jpg" | wc -l`, which should output 109,554 in your terminal.

As for the annotations from our SUN-SGE, you are happy to execute:

- Unwarp it via `unzip SUN-SEG-Annotation-v2.zip`
- Put it at path `./data/SUN-SEG-Annotation/` 

After preparing all the files, your file structure will be the same as below:

```
├──data
    ├──SUN-Positive
        ├──case1
            ├──image_name.jpg
            |...
        ├──case2
        |...
    ├──SUN-SEG-Annotation
        ├──TrainDataset
            ├──GT
                ├──case1_1
                    ├──image_name.png
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
            ├──Seen
                ├──GT
                |...
            ├──Unseen
                ├──GT
                |...
        ├──TestHardDataset
            ├──Seen
                ├──GT
                |...
            ├──Unseen
                ├──GT
                |...
```

You will notice that the file structure of images in `SUN-Positive` is different from the one of annotation in `SUN-SEG-Annotation`. To reconcile the file structure, you need to follow next step to finish the data preparation.

# Step-3: Re-organize the file structure

By running `sh ./utils/reorganize.sh`, the original file structure in SUN-dataset will be reorganised to the same as SUN-SEG for better length balance. Finally, the folder `Frame` which is originated from `SUN-Positive`, and `GT`, as long as other annotations folders, will share the same file structure as shown below:

```
├──data
    ├──SUN-SEG
        ├──TrainDataset
            ├──Frame  # The images from the SUN dataset
                ├──case1_1
                    ├──image_name_00001.jpg
                    |...
                ├──case1_3
                |...
            ├──GT  # Object-level segmentation mask
                ├──case1_1
                    ├──image_name_00001.png
                    |...
                ├──case1_3
                |...
            ├──Edge  # Weak label with edge
                |...
            ├──Scribble  # Weak label with scribble
                |...
            ├──Polygon  # Weak label with Polygon
                |...
            ├──Classification  # Category classification annotation
                ├──classification.txt
            ├──Detection  # Bounding box
                ├──bbox_annotation.json
        ├──TestEasyDataset
            ├──Seen
                ├──Frame
                    ├──case2_3
                    |...
                ├──GT
                    ├──case2_3
                    |...
                |...
            ├──Unseen
                ├──Frame
                    ├──case3_1
                    |...
                ├──GT
                    ├──case3_1
                    |...
                |...
        ├──TestHardDataset
            ├──Seen
                ├──Frame
                    ├──case1_2
                    |...
                ├──GT
                    ├──case1_2
                    |...
                |...
            ├──Unseen
                ├──Frame
                    ├──case10_1
                    |...
                ├──GT
                    ├──case10_1
                    |...
                |...
```
