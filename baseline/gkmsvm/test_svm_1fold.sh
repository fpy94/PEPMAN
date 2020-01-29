for i in `seq 0 9`
do
./lsgkm/bin/gkmpredict -T 16 ../../data/HeLaS3_positive_test.fa ./1_fold_model/1_fold_$i.model.txt ./1_fold_test/test_pos_1fold_$i.txt
./lsgkm/bin/gkmpredict -T 16 ./HeLaS3_negative_test_1fold.fa ./1_fold_model/1_fold_$i.model.txt ./1_fold_test/test_neg_1fold_$i.txt
done
#./lsgkm/bin/gkmpredict -T 16 ../../../../4.training_data_all_4_1/HeLaS3_positive_test.fa ./10_fold.model.txt test_pos_10fold.txt
#./lsgkm/bin/gkmpredict -T 16 ../../../../4.training_data_all_4_1/HeLaS3_negative_test.fa ./10_fold.model.txt test_neg_10fold.txt
