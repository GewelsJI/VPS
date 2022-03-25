# The candidate competitors are listed here:
# MODEL_NAMES=('2022-TMI-PNSPlus' '2021-NIPS-AMD' '2021-MICCAI-PNSNet' '2021-ICCV-FSNet' '2021-ICCV-DCFNet' '2020-TIP-MATNet' '2020-MICCAI-23DCNN' '2020-AAAI-PCSA' '2019-TPAMI-COSNet' '2021-TPAMI-SINetV2' '2021-MICCAI-SANet' '2020-MICCAI-PraNet' '2020-MICCAI-ACSNet' '2018-TMI-UNet++' '2015-MICCAI-UNet')

MODEL_NAMES=('2020-MICCAI-23DCNN')

for MODEL_NAME in ${MODEL_NAMES[*]}
do
  nohup python -u vps_evaluator.py --data_lst TestEasyDataset --model_lst $MODEL_NAME  --txt_name $MODEL_NAME >> ./loggings/$MODEL_NAME-1.log &
  nohup python -u vps_evaluator.py --data_lst TestHardDataset --model_lst $MODEL_NAME  --txt_name $MODEL_NAME >> ./loggings/$MODEL_NAME-2.log &
done
