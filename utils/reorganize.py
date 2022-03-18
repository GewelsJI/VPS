import os, shutil, glob

SUN_root = './data/SUN-Positive/'
SUNSEG_root = './data/SUN-SEG-Annotation/'

SUN_split_dict = {}
SUNSEG_split_dict = {}
SUNSEG_dataset_dict = {}
image_list = []

SUN_list = glob.glob(SUN_root+'*/*.jpg')
SUNSEG_list = glob.glob(SUNSEG_root+'*/GT/*/*.png')

# Get SUN and SUN-SEG case-to-image structure in a dictionary
for SUN_path, SUNSEG_path in zip(SUN_list, SUNSEG_list):

    SUN_case_name, SUN_image_name = SUN_path.split('/')[-2], SUN_path.split('/')[-1].rstrip('.jpg')
    SUNSEG_dataset_name, SUNSEG_case_name, SUNSEG_image_name = SUNSEG_path.split('/')[-4], SUNSEG_path.split('/')[-2], SUNSEG_path.split('/')[-1].rstrip('.png')

    SUN_split_dict[SUN_image_name] = SUN_case_name
    SUNSEG_split_dict[SUNSEG_image_name] = SUNSEG_case_name
    SUNSEG_dataset_dict[SUNSEG_image_name] = SUNSEG_dataset_name

    image_list.append(SUN_image_name)

# Change original SUN's structure
for image in image_list:
    SUN_case = SUN_split_dict[image]
    SUNSEG_case = SUNSEG_split_dict[image]
    dataset_split = SUNSEG_dataset_dict[image]

    os.makedirs(os.path.join(SUNSEG_root, dataset_split, 'Frame', SUNSEG_case), exist_ok=True)

    shutil.move(os.path.join(SUN_root, SUN_case, image + '.jpg'),
                os.path.join(SUNSEG_root, dataset_split, 'Frame', SUNSEG_case, image + '.jpg'))



