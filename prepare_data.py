from Bio import SeqIO
import numpy as np
import pickle

celline='HeLaS3'
pos_train = './data/'+celline+'_positive_train.fa'
pos_test = './data/'+celline+'_positive_test.fa'
pos_valid = './data/'+celline+'_positive_valid.fa'
neg_train = './data/'+celline+'_negative_train.fa'
neg_test = './data/'+celline+'_negative_test.fa'
neg_valid = './data/'+celline+'_negative_valid.fa'

onehot_dic = {
    "A": np.array([[1,0,0,0]], dtype="float32"),
    "a": np.array([[1,0,0,0]], dtype="float32"),
    "T": np.array([[0,1,0,0]], dtype="float32"),
    "t": np.array([[0,1,0,0]], dtype="float32"),
    "G": np.array([[0,0,1,0]], dtype="float32"),
    "g": np.array([[0,0,1,0]], dtype="float32"),
    "C": np.array([[0,0,0,1]], dtype="float32"),
    "c": np.array([[0,0,0,1]], dtype="float32"),
    "n": np.array([[0.25,0.25,0.25,0.25]], dtype="float32"),
    "N": np.array([[0.25,0.25,0.25,0.25]], dtype="float32"),
    "=": np.array([[0.25,0.25,0.25,0.25]], dtype="float32")}

def process_fasta(filename):
    tmp = []
    for seq_record in SeqIO.parse(filename,"fasta"):
        seq = seq_record.seq
        length = len(seq)
        positive_seq = []
        for char in seq:
            if char in onehot_dic:
                positive_seq.append(onehot_dic[char])
        positive_seq = np.array(positive_seq)
        positive_seq = positive_seq.reshape([length,4])
#    positive_seq = positive_seq.T
        tmp.append(positive_seq)
    return tmp

pos_train_vec = np.array(process_fasta(pos_train))
pos_test_vec = np.array(process_fasta(pos_test))
pos_valid_vec = np.array(process_fasta(pos_valid))

neg_train_vec = np.array(process_fasta(neg_train))
neg_test_vec = np.array(process_fasta(neg_test))
neg_valid_vec = np.array(process_fasta(neg_valid))

pos_train_num = len(pos_train_vec)
pos_test_num = len(pos_test_vec)
pos_validation_num = len(pos_valid_vec)
neg_train_num = len(neg_train_vec)
neg_test_num = len(neg_test_vec)
neg_validation_num = len(neg_valid_vec)

X_pos_train = pos_train_vec
X_neg_train = neg_train_vec

X_pos_test = pos_test_vec
X_neg_test = neg_test_vec

X_pos_valid = pos_valid_vec
X_neg_valid = neg_valid_vec
fp = open('./data/data.pkl', 'wb')
data = {'X_train_pos':np.array(X_pos_train),'X_train_neg':np.array(X_neg_train),'X_test_pos':np.array(X_pos_test), 'X_test_neg':np.array(X_neg_test),'X_valid_pos':np.array(X_pos_valid), 'X_valid_neg':np.array(X_neg_valid)}
pickle.dump(data,fp)
