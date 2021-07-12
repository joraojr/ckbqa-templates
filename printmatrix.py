import pandas as pd
from pretty_print_cf import pretty_plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np

if __name__ == '__main__':
    result = pd.read_csv(
        "analysis-lcquad-2-wikidata/expname=with--attention--3,input_dim=389,mem_dim=150,lr=0.01,emblr=0.01,wd=0.00225,epochs=100,current_epoch=38,test_acc=0.722,loss=0.8938479226351083.csv",
        header=None)
    encoder = LabelEncoder()
    encoder.classes_ = np.load(
        '/Users/joraojr./Documents/Mestrado/ckbqa-templates/data/lc-quad-2-wikidata/le_dummy.npy')
    class_names = encoder.classes_
    print(class_names)
    y_true = result[0].values
    y_pred = result[1].values
    matrix = confusion_matrix(y_true, y_pred)
    print(matrix)
    df_cm = pd.DataFrame(matrix, index=class_names, columns=class_names)
    plt = pretty_plot_confusion_matrix(df_cm, figsize=[80, 80], fz=55)
    plt.savefig('confusion_matrix_wikidata_attention.png')
    # plt.show()
