import os, shutil, glob

SUN_root = './data/SUN-Positive/'
SUNSEG_root = './data/SUN-SEG-Annotation/'

SUN_split_dict = {}
SUNSEG_split_dict = {}
SUNSEG_dataset_dict = {}
image_list = []

# SUN_list = glob.glob(SUN_root + '*/*.jpg')
SUNSEG_test_list = glob.glob(SUNSEG_root + 'Test*/*/GT/*/*.png')
SUNSEG_train_list = glob.glob(SUNSEG_root + 'TrainDataset/GT/*/*.png')
SUNSEG_list = SUNSEG_test_list + SUNSEG_train_list

SUN_list = [os.path.join(SUN_root, name.split('/')[-2].split('_')[0] if len(name.split('/')[-2].split('_')) > 1 else name.split('/')[-2], name.split('/')[-1].replace('.png', '')) for name in SUNSEG_list]

for SUN_path, SUNSEG_path in zip(SUN_list, SUNSEG_list):
    """
        @func: Get SUN and SUN-SEG case-to-image structure in a dictionary
    """

    SUN_case_name, SUN_image_name = SUN_path.split('/')[-2], SUN_path.split('/')[-1]
    SUNSEG_dataset_name, SUNSEG_case_name, SUNSEG_image_name = SUNSEG_path.split('SUN-SEG-Annotation/')[1].split('/GT')[0], SUNSEG_path.split('/')[-2], SUNSEG_path.split('/')[-1].rstrip('.png')

    SUN_split_dict[SUN_image_name] = SUN_case_name
    SUNSEG_split_dict[SUNSEG_image_name] = SUNSEG_case_name
    SUNSEG_dataset_dict[SUNSEG_image_name] = SUNSEG_dataset_name

    image_list.append(SUN_image_name)

for image in image_list:
    """
        @func: Change original SUN's structure
    """
    SUN_case = SUN_split_dict[image]
    SUNSEG_case = SUNSEG_split_dict[image]
    dataset_split = SUNSEG_dataset_dict[image]

    os.makedirs(os.path.join(SUNSEG_root, dataset_split, 'Frame', SUNSEG_case), exist_ok=True)

    shutil.move(os.path.join(SUN_root, SUN_case, image + '.jpg'),
                os.path.join(SUNSEG_root, dataset_split, 'Frame', SUNSEG_case, image + '.jpg'))
