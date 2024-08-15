
# CS728 Assignment 2

  

**Goda Nagakalyani (214050010)**

This file contains instruction on how to run the code and structure of the code.

  

## Requirement
*Execute All the commands from inside submit folder.
Install required libraries by running the following command,

  

```bash

pip install  -r  requirements.txt  //  for  pip

conda install  --file  requirements.txt  //  for  conda

```

  

## Folder Structure

submit/

>code/

>>Q1.py // Contains both the model architecture, training and testing for Q1

>>DTW.py // Contains the Model for DTW non_crossing and crossing

>>Q3.py // Code for Training and testing of DTW models.

>>dataset.py // Dataset class for Glue dataset.

  

>output/

>> Q1_test.txt // Finetuned model predicted output for Test dataset of GLUE.

>> Q1_val.txt // Finetuned model predicted output for Validation dataset of GLUE.

>> Q3_non_crossing_test.txt // DTW non Crossing model predicted output for Test dataset of GLUE.

>> Q3_non_crossing_val.txt // DTW non Crossing model predicted output for Val dataset of GLUE.

>> Q3_crossing_test.txt // DTW Crossing model predicted output for Test dataset of GLUE.

>> Q3_crossing_val.txt // DTW Crossing model predicted output for Test dataset of GLUE.

  

>additional_files/

>>models/ // Contains save parameters for all the model (Q1,Q2,Q3).

>README.md

>Requirements.txt

>Report.pdf

## Method to reproduce the result and test the model

### Q1

#### Training

To fine tune the the BERT-Tiny model with BST task. Use the following command.

* Note this will overwrite the existing model params in additional folder. To only test, use the commands in the next segment.

```bash

python3 code/Q1.py -t

```

This will also generate results for validation and test dataset in "*outputs/*" directory

  

#### Inference

To test/infer the model use the following command:

```bash

python3 code/Q1.py -i

```

This will give the user a prompt to enter sentence 1 and sentence 2 respectively and It will show the Correlation score.

  

### Q3

#### Training

To train a,b of tanh in DTW with **non crossing** constraint, run the following command.

* Note this will overwrite the existing model params in additional folder. To only test, use the commands in the next segment.

```bash

python3 code/Q3.py -t

```

To train a,b of tanh in DTW with **crossing** constraint, add another arg "-c" to the above command.

```bash

python3 code/Q3.py -c -t

```

This will also generate results for validation and test dataset in "*outputs/*" directory with the names of the files as *{DTW_{non_crossing/crossing}_{val/test}.txt}*

#### Inference

To test/infer the model use the following command:

```bash

python3 Q3.py -i

```

Similar to training add "-c" for inference using crossing in above command.

This will also give the user a prompt to enter sentence 1 and sentence 2 respectively and It will show the Correlation score and a mapping between the tokens of smaller sentence with the larger sentence.
