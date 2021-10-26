import os

import metriculous

import pandas as pd
import sklearn.metrics

from pretty_print_cf import pretty_plot_confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn import metrics

# if __name__ == "__main__":
#     result = pd.read_csv(
#         "analysis-lcquad-2-wikidata/expname=with--attention--3,input_dim=389,mem_dim=150,lr=0.01,emblr=0.01,wd=0.00225,epochs=100,current_epoch=38,test_acc=0.722,loss=0.8938479226351083.csv",
#         header=None)
#     encoder = LabelEncoder()
#     encoder.classes_ = np.load(
#         '/Users/joraojr./Documents/Mestrado/ckbqa-templates/data/lc-quad-2-wikidata/le_dummy.npy')
#     class_names = encoder.classes_
#
#     #    y_true = np.reshape(result[0].values, (1, len(result[0].values)))
#     #    y_pred = np.reshape(result[1].values, (1, len(result[1].values)))
#
#     y_true = result[0].values
#     y_pred = np.eye(len(class_names))[result[1].values]
#     print(y_true)
#     print(y_pred)
#     print(y_true.shape)
#     print(y_pred.shape)
#
#     metriculous.compare_classifiers(
#         ground_truth=y_true,
#         model_predictions=[y_pred],
#         model_names=["Perfect Model"],
#         # class_names=class_names
#     ).save_html("/Users/joraojr./Documents/Mestrado/ckbqa-templates/comparison.html")


# final_results folder
melhores = {
    "HTL": "expname=HDT,input_dim=387,mem_dim=150,lr=0.01,emblr=0.01,wd=0.00225,epochs=100,current_epoch=6,test_acc=0.7083761025318231,loss=0.7950990411337784.csv",
    "HTL_paraphase": "expname=HDT_parafrase,input_dim=387,mem_dim=150,lr=0.01,emblr=0.01,wd=0.00225,epochs=100,current_epoch=26,test_acc=0.8914439378944784,loss=0.44518669838021924.csv",
    "HTL_group": "expname=HDT_group,input_dim=387,mem_dim=150,lr=0.01,emblr=0.01,wd=0.00225,epochs=100,current_epoch=5,test_acc=0.8479222473593585,loss=0.4420519198290738.csv",
    "HTL_group_parafrase": "expname=HDT_group_parafrase,input_dim=387,mem_dim=150,lr=0.01,emblr=0.01,wd=0.00225,epochs=100,current_epoch=19,test_acc=0.9276748811080627,loss=0.3077197893595488.csv",
    "TREE_LSTM": "expname=lcquad_1-false,input_dim=385,mem_dim=150,lr=0.01,emblr=0.01,wd=0.00225,epochs=100,current_epoch=24,test_acc=0.8174273858921162,loss=0.7823939571962588.csv",
    "Tree": "expname=TREE,input_dim=387,mem_dim=150,lr=0.01,emblr=0.01,wd=0.00225,epochs=100,current_epoch=5,test_acc=0.728834975991913,loss=0.8109815468622815.csv",
    "Tree_group": "expname=TREE_group,input_dim=387,mem_dim=150,lr=0.01,emblr=0.01,wd=0.00225,epochs=100,current_epoch=3,test_acc=0.8364922921405105,loss=0.48486261481087806.csv",
    "Tree_paraphase": "expname=TREE_paraphase,input_dim=387,mem_dim=150,lr=0.01,emblr=0.01,wd=0.00225,epochs=100,current_epoch=40,test_acc=(0.8915845337376801, 0.8893044487620614),loss=0.46058784208078063.csv",
    "Tree_group_paraphase": "expname=TREE_group_paraphase,input_dim=387,mem_dim=150,lr=0.01,emblr=0.01,wd=0.00225,epochs=100,current_epoch=14,test_acc=(0.9214051048774324, 0.9197940185614562),loss=0.30137250077024125.csv"
}
if __name__ == "__main__":
    #    expname=HDT,input_dim=387,mem_dim=150,lr=0.01,emblr=0.01,wd=0.00225,epochs=100,current_epoch=7,test_acc=0.7163117581169844,loss=0.7938936256284359.csv",

    abc = melhores["Tree_group_paraphase"]
    file = "./final_results/analysis/Tree_group_paraphase/" + abc  # expname=HDT,input_dim=387,mem_dim=150,lr=0.01,emblr=0.01,wd=0.00225,epochs=100,current_epoch=50,test_acc=0.7170883488208845,loss=1.4860947186505116.csv"
    result = pd.read_csv(file, header=None)
    encoder = LabelEncoder()
    encoder.classes_ = np.load(
        '/Users/joraojr./Documents/Mestrado/ckbqa-templates/data/lc-quad-2-wikidata-parafrase/le_dummy.npy')
    class_names = encoder.classes_
    class_names = [str(name) for name in class_names]
    print(class_names)
    #    y_true = np.reshape(result[0].values, (1, len(result[0].values)))
    #    y_pred = np.reshape(result[1].values, (1, len(result[1].values)))

    y_pred = result[0].values
    y_true = result[1].values

    print(metrics.classification_report(y_true, y_pred,
                                        digits=3))  # target_names=class_names))#, labels=[13,16,17,18,19,20,21,23,24,26,27]))  # target_names=class_names))

    #    print(metrics.precision_score(y_true, y_pred,average="macro"))
    #    print(metrics.recall_score(y_true, y_pred,average="macro"))
    #    print(metrics.f1_score(y_true, y_pred,average="macro"))

    print(metrics.balanced_accuracy_score(y_true, y_pred))
    print(metrics.accuracy_score(y_true, y_pred))

# if __name__ == "__main__":
#
#     dir = "./final_results/analysis/Tree_group_paraphase/"
#     for filename in os.listdir(dir):
#         if filename == ".DS_Store":
#             continue
#         file = os.path.join(dir, filename)
#         result = pd.read_csv(file, header=None)
#         y_pred = result[0].values
#         y_true = result[1].values
#         names = file.split(",")
#         for i, n in enumerate(names):
#             if "test_acc=" in n:
#                 names[
#                     i] = f"test_acc=({metrics.accuracy_score(y_true, y_pred)},{metrics.balanced_accuracy_score(y_true, y_pred)})"
#                 break
#             elif "dev_acc=" in n:
#                 names[
#                     i] = f"dev_acc=({metrics.accuracy_score(y_true, y_pred)},{metrics.balanced_accuracy_score(y_true, y_pred)})"
#                 break
#         print(",".join(names))
