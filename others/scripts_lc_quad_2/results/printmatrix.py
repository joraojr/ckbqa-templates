import pandas as pd
from pretty_print_cf import pretty_plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np

if __name__ == '__main__':
    # result = pd.read_csv("final_results/analysis/HDT_group/expname=HDT_group,input_dim=387,mem_dim=150,lr=0.01,emblr=0.01,wd=0.00225,epochs=100,current_epoch=5,test_acc=0.8479222473593585,loss=0.4420519198290738.csv",
    result = pd.read_csv(
        "final_results/analysis/HDT/expname=HDT,input_dim=387,mem_dim=150,lr=0.01,emblr=0.01,wd=0.00225,epochs=100,current_epoch=6,test_acc=0.7083761025318231,loss=0.7950990411337784.csv",
        header=None)
    encoder = LabelEncoder()
    encoder.classes_ = np.load(
        '/Users/joraojr./Documents/Mestrado/ckbqa-templates/data/lc-quad-2-wikidata-parafrase/le_dummy.npy')
    class_names = encoder.classes_
    print(class_names)
    y_true = result[1].values
    y_pred = result[0].values
    matrix = confusion_matrix(y_true, y_pred)
    print(matrix)
    df_cm = pd.DataFrame(matrix, index=class_names, columns=class_names)
    plt = pretty_plot_confusion_matrix(df_cm, figsize=[60, 60], fz=40, cmap="Blues")
    plt.savefig('matrix_template.png')
    # plt.show()
