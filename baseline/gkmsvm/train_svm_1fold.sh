for i in `seq 0 9`
do
./lsgkm/bin/gkmtrain -T 16 -m 16384.0 ../../data/HeLaS3_positive_train.fa ./train_data/HeLaS3_negative_train_1fold_$i.fa ./1_fold_model/1_fold_$i
done
