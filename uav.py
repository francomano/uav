#Classification:total number of conflict 36th column (num_collisions)
#regression:min_CPA 37th (min_CPA)
#[0,1,2,3,4] dataset unbalanced
#NORMALIZE DATA
#TRY DIFFERENT HYPERPARAMETERS(GRID)
#TRY ENSAMBLE (one with bayes inside)
#naive bayas works well on indipendent features (maybe this case)
#svm with outliers(try to plot something)



#sklearn accept pandas dataframe as input
###########PANDA SNIPPET###############
import pandas as pd
dataset=pd.read_csv("train_set.tsv",sep='\t',header=0)

X=dataset.iloc[:,:-2]  # : all raws :-2 slice out the last 2 columns
y=dataset.iloc[:,-2]   #just the last column
yr=dataset.iloc[:,-1]

uav_1_x=X['UAV_1_x']
#print(yr)


#normalized_df=(dataset-dataset.mean())/dataset.std()
normalized_df=(X-X.min())/(X.max()-X.min())  #between 0 and 1 (is column-wise)

 ws=sklearn.utils.class_weight.compute_class_weight(class_weight='balanced', classes=np.array([0,1,2,3,4]), y=y)


 #grid-search for hyperparameters of kernel: to much time