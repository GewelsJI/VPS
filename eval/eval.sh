MODEL_NAME='2022-TMI-PNSPlus'
nohup python -u vps_evaluator.py --data_lst TestEasyDataset --model_lst $MODEL_NAME  --txt_name $MODEL_NAME >> ./loggings/$MODEL_NAME-1.log &
nohup python -u vps_evaluator.py --data_lst TestHardDataset --model_lst $MODEL_NAME  --txt_name $MODEL_NAME >> ./loggings/$MODEL_NAME-2.log &

MODEL_NAME='2021-NIPS-AMD'
nohup python -u vps_evaluator.py --data_lst TestEasyDataset --model_lst $MODEL_NAME  --txt_name $MODEL_NAME >> ./loggings/$MODEL_NAME-1.log &
nohup python -u vps_evaluator.py --data_lst TestHardDataset --model_lst $MODEL_NAME  --txt_name $MODEL_NAME >> ./loggings/$MODEL_NAME-2.log &

MODEL_NAME='2021-MICCAI-PNSNet'
nohup python -u vps_evaluator.py --data_lst TestEasyDataset --model_lst $MODEL_NAME  --txt_name $MODEL_NAME >> ./loggings/$MODEL_NAME-1.log &
nohup python -u vps_evaluator.py --data_lst TestHardDataset --model_lst $MODEL_NAME  --txt_name $MODEL_NAME >> ./loggings/$MODEL_NAME-2.log &

MODEL_NAME='2021-ICCV-FSNet'
nohup python -u vps_evaluator.py --data_lst TestEasyDataset --model_lst $MODEL_NAME  --txt_name $MODEL_NAME >> ./loggings/$MODEL_NAME-1.log &
nohup python -u vps_evaluator.py --data_lst TestHardDataset --model_lst $MODEL_NAME  --txt_name $MODEL_NAME >> ./loggings/$MODEL_NAME-2.log &

MODEL_NAME='2021-ICCV-DCFNet'
nohup python -u vps_evaluator.py --data_lst TestEasyDataset --model_lst $MODEL_NAME  --txt_name $MODEL_NAME >> ./loggings/$MODEL_NAME-1.log &
nohup python -u vps_evaluator.py --data_lst TestHardDataset --model_lst $MODEL_NAME  --txt_name $MODEL_NAME >> ./loggings/$MODEL_NAME-2.log &

MODEL_NAME='2020-TIP-MATNet'
nohup python -u vps_evaluator.py --data_lst TestEasyDataset --model_lst $MODEL_NAME  --txt_name $MODEL_NAME >> ./loggings/$MODEL_NAME-1.log &
nohup python -u vps_evaluator.py --data_lst TestHardDataset --model_lst $MODEL_NAME  --txt_name $MODEL_NAME >> ./loggings/$MODEL_NAME-2.log &

MODEL_NAME='2020-MICCAI-23DCNN'
nohup python -u vps_evaluator.py --data_lst TestEasyDataset --model_lst $MODEL_NAME  --txt_name $MODEL_NAME >> ./loggings/$MODEL_NAME-1.log &
nohup python -u vps_evaluator.py --data_lst TestHardDataset --model_lst $MODEL_NAME  --txt_name $MODEL_NAME >> ./loggings/$MODEL_NAME-2.log &

MODEL_NAME='2020-AAAI-PCSA'
nohup python -u vps_evaluator.py --data_lst TestEasyDataset --model_lst $MODEL_NAME  --txt_name $MODEL_NAME >> ./loggings/$MODEL_NAME-1.log &
nohup python -u vps_evaluator.py --data_lst TestHardDataset --model_lst $MODEL_NAME  --txt_name $MODEL_NAME >> ./loggings/$MODEL_NAME-2.log &

MODEL_NAME='2019-TPAMI-COSNet'
nohup python -u vps_evaluator.py --data_lst TestEasyDataset --model_lst $MODEL_NAME  --txt_name $MODEL_NAME >> ./loggings/$MODEL_NAME-1.log &
nohup python -u vps_evaluator.py --data_lst TestHardDataset --model_lst $MODEL_NAME  --txt_name $MODEL_NAME >> ./loggings/$MODEL_NAME-2.log &

MODEL_NAME='2021-TPAMI-SINetV2'
nohup python -u vps_evaluator.py --data_lst TestEasyDataset --model_lst $MODEL_NAME  --txt_name $MODEL_NAME >> ./loggings/$MODEL_NAME-1.log &
nohup python -u vps_evaluator.py --data_lst TestHardDataset --model_lst $MODEL_NAME  --txt_name $MODEL_NAME >> ./loggings/$MODEL_NAME-2.log &

MODEL_NAME='2021-MICCAI-SANet'
nohup python -u vps_evaluator.py --data_lst TestEasyDataset --model_lst $MODEL_NAME  --txt_name $MODEL_NAME >> ./loggings/$MODEL_NAME-1.log &
nohup python -u vps_evaluator.py --data_lst TestHardDataset --model_lst $MODEL_NAME  --txt_name $MODEL_NAME >> ./loggings/$MODEL_NAME-2.log &

MODEL_NAME='2020-MICCAI-PraNet'
nohup python -u vps_evaluator.py --data_lst TestEasyDataset --model_lst $MODEL_NAME  --txt_name $MODEL_NAME >> ./loggings/$MODEL_NAME-1.log &
nohup python -u vps_evaluator.py --data_lst TestHardDataset --model_lst $MODEL_NAME  --txt_name $MODEL_NAME >> ./loggings/$MODEL_NAME-2.log &

MODEL_NAME='2020-MICCAI-ACSNet'
nohup python -u vps_evaluator.py --data_lst TestEasyDataset --model_lst $MODEL_NAME  --txt_name $MODEL_NAME >> ./loggings/$MODEL_NAME-1.log &
nohup python -u vps_evaluator.py --data_lst TestHardDataset --model_lst $MODEL_NAME  --txt_name $MODEL_NAME >> ./loggings/$MODEL_NAME-2.log &

MODEL_NAME='2018-TMI-UNet++'
nohup python -u vps_evaluator.py --data_lst TestEasyDataset --model_lst $MODEL_NAME  --txt_name $MODEL_NAME >> ./loggings/$MODEL_NAME-1.log &
nohup python -u vps_evaluator.py --data_lst TestHardDataset --model_lst $MODEL_NAME  --txt_name $MODEL_NAME >> ./loggings/$MODEL_NAME-2.log &

MODEL_NAME='2015-MICCAI-UNet'
nohup python -u vps_evaluator.py --data_lst TestEasyDataset --model_lst $MODEL_NAME  --txt_name $MODEL_NAME >> ./loggings/$MODEL_NAME-1.log &
nohup python -u vps_evaluator.py --data_lst TestHardDataset --model_lst $MODEL_NAME  --txt_name $MODEL_NAME >> ./loggings/$MODEL_NAME-2.log &