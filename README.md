# IRDM17
IRDM project


###### Data folder
In the data folder, one will find the scripts to merge the data_csv file into one big file. Applying the stemming process to all concerned columns and spell checking. The results are then saved into different pickles file. Those will then be used in the feature folder to create the featres.

The features folder contains the diffrent script to pre-compute the different features and save them into one file.






#####Model folder

In this folder you will find the procedure to test the different model. WideAndDeepModel.ipynb is a variant opf the cross_validation_procedure.

This variant is designed to test various neural network models with different parameters.

The cross_validaiton_procedure is the main one used to test all the model (cf group report). 

You can see that some models have been comented out, to test different model just changed which model is commented out in the train model section.

Both WideAndDeepModel.ipynb  and the cross_validation give the exam same input and take as an output the same input. 

The picle object "the_matrix_3feat.p" and others are the input we can use for the models. They are the output of the other section preparing the data
and computing the feature, and combining them into one matrix. We have three variant including all to zero of Ulysses features. This because
they were to heave for some pre-analysis (with same results) so we didn't include them in all (cf group report). 

To test the model you can both change the model commented out and the number of Ulysses feature used. 
