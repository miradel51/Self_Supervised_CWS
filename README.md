# Self_Supervised_CWS

Segment, Mask, and Predict: Augmenting Chinese Word Segmentation with Self-Supervision

Please cite:

```
@inproceedings{maimaiti-etal-2021-segment,
    title = "Segment, Mask, and Predict: Augmenting {C}hinese Word Segmentation with Self-Supervision",
    author = "Maimaiti, Mieradilijiang  and
      Liu, Yang  and
      Zheng, Yuanhang  and
      Chen, Gang  and
      Huang, Kaiyu  and
      Zhang, Ji  and
      Luan, Huanbo  and
      Sun, Maosong",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.158",
    pages = "2068--2077",
}
```

The following parts of this project are being prepared gradually.

## Datasets

All the corpora used in our experiment are from SIGHAN05, SIGHAN08, SIGHAN10 and some OTHER open datasets respectively.

- **SIGHAN05**
    - It contains "MSRA, PKU, AS and CITYU" corpora.
    - The related [paper](https://aclanthology.org/I05-3017.pdf) and [datasets](http://sighan.cs.uchicago.edu/bakeoff2005/) 

- **SIGHAN08/10**
    - As sighan2008 and sighan2010 corpora are proprietary, we are unable to distribute them. If you have a legal copy, you can replicate our scores following the same codes as the other corpora.

- **OTHER**
    - It contains "CNC, UDC and ZX" corpora.
    - The related [paper](https://aclanthology.org/E14-1062.pdf) and [datasets](https://github.com/hankcs/multi-criteria-cws/tree/master/data/other/) 
    

## Requirements
- Preprocessing
    - We use the same data pre-processing as used in [Huang et al. (2020)](https://aclanthology.org/2020.emnlp-main.318.pdf)
    - We use the original format of AS and CITYU instead of using their corresponding simplified versions.
    - Other related scripts are given in `preprocess` folder. 
- Environment (necessary)
    - Python 3.6
    - torch>=1.4.0
    - transformers>=4.4.2

## Usage
- Preprocessing
    - Cleaning data: we clean the data by using simple four steps instead of dirctly train the model using openly available corpus.
        
        > Step1, remove the continuous multiple white spaces and replace them into single white space. 
        
        ```
        python replace_sp.py original_file rm_sp_file
        ```
        
        > Step2, remove the blank lines among the corpora.
        
        ```
        python rmv_blk_ln.py rm_sp_file rm_blk_file
        ```
        
        > Step3, remove the duplicated lines from the corpora, if there are existed some identical lines.
        
        ```
        python rmv_dup.py rm_blk_file rm_dp_file
        ```
        
        > Step4, shuffle the copurs (trainset and devset only) after the previous three steps. 

        ```
        python shuffle_corpus.py --corpus rm_dp_file
        ```
        
    - Generating nosiy data: The noisy data generation consists of two steps. 
    
        > First, convert the sequence into char sequence. Second, generate "BMES" 4 classes randomly.
    
        step1, run `con_char.py` to achieve the char sequence.
        ```
        python con_char.py original_file char_file
        ```
    
        step2, run `gen_4class_rnd.py` to generate the randomly segmented files based on char sequence that achieved from `step1`.
        ```
        python gen_4class_rnd.py char_file random_labeled_file 
        ```
     - Cutting the corpus according to the punctuation
        ```
        python cut.py input_file output_file
        ```
     - Convert both the trainset/devset and test file into the same format as [EMNLP2020](https://aclanthology.org/2020.emnlp-main.318.pdf). The digits are converted into '0', and the alphabets are converted into 'X'. The converting script `convert_fomat_utils.py` is in the `preprocess` folder.

        > Convert the trainset/devset:
        ```
        python convert_fomat_utils.py -f p-train -i input_file -o output_file
        ```

        > Convert the test file:
        ```
        python convert_fomat_utils.py -f p-test -i input_file -o output_file
        ```

     - Building vocab file

        > Training the MLM requires a vocabulary file. A sample vocabulary file is located at `train/mlm/mydata/vocab.txt`. If you want to use your own vocabulary, you may use the script `build_vocab.py` in the `preprocess` folder (which is originated from [THUMT](https://github.com/THUNLP-MT/THUMT)).

        ```
        python build_vocab.py input_file vocab_file
        ```

        > Note: (1) the input file should be converted into `char` format in this step. (2) After generating the vocab file using the aforementioned command, you should replace the first 3 rows in the vocab file with the content of the file `V1.txt` in the `preprocess` folder.
        
- Training
    
    > All the training process includes xx steps such as, training the MLM, Predicor, MRT risk file generation, Segmenter training, Optimizing segmenter using MRT. Most of corresponding codes in the train folder. Besides, in the training of segmenter we highgly follow their [Huang et al. (2020)](https://aclanthology.org/2020.emnlp-main.318.pdf) code. Therefore, in the preprocessing step we convert the original data into requited format.
    
    - MLM: train the revised masked language model for the predictor using new masking strategy. 
        > Hint: please set the available GPU number for training the MLM, and you can set the masked number by revising the parameter mask_count. Besides, to achieve the better result we combine the `SIGHAN05, SIGHAN08 and OTHER` as the dataset. Then split them train file and test file. The corpus should be `cut` according to the punctuation before training the MLM. Corresponding code in `train/mlm` folder.
    
     ```
     sh run_mlm_scratch.sh
     ```
    
    - Predictor: predict the maksed position using revised masked language model in the input sequence.
        > You may also need to choose the available GPU number for predicting the masked position in the train set using previously trained revised MLM. Moreover, the corpus should be converted into `char` format in this step. The corresponding code in `train/predictor` forlder.
    
     ```
     sh pred_mlm_mrt.sh
     ```
    
    - Revised MRT: generate some related files for revised MRT, the original paper for MRT in [here](https://aclanthology.org/P16-1159.pdf).
        > In this step, you need to build the neccessary file `train.mrt` and `train.risk` which will used in optimizing the segmenter via MRT. Besides, both the predicted result `pred_mlm.txt` achieved by `Predictor` and the trainset `train.cut` are required. The parameter `MASK_COUNT` should keep the consistent with the training `MLM` and `Predictor`, while you can also set the mrt size parameter `N_MRT` (default is 10). Corresponding code in `train/revised_mrt` folder.
    
    ```
    python make_mrt.py
    ```
    
    - Segmenter: train the segmenter with CRF. 
        > The segmenter part is revised based on the original code of  [Huang et al. (2020)](https://aclanthology.org/2020.emnlp-main.318.pdf). The trainset is `train.cut` of the corresponding corpora, and both the trainset and devset are converted into the same format as  [Huang et al. (2020)](https://aclanthology.org/2020.emnlp-main.318.pdf). While you can set the GPU number with the parameter `CUDA_ENV_NUM` and the corrosponding codes are in `train/segmenter` folder.
    
    ```
    python main_crf.py -c config_crf.txt
    ```
    - Optimizing segmenter:  continue to train the `Segmenter` using MRT.
        > In this step you may need to use the generated mrt files that achieved from `Revised MRT` step. Besides, the devset is also need to be converted into the same format as [Huang et al. (2020)](https://aclanthology.org/2020.emnlp-main.318.pdf). Moreoevr, it is neccesary to keep the consistent with the same mrt size of  `Revised MRT` step, while you can also revise the other hyper-parameters for MRT such as lambda and alpha via `reg_lambda` and `alpha_mrt`, respectively. The corrosponding codes are in `train/segmenter` folder as well.
    
     ```
     python main_mrt_crf.py -c config_mrt_crf.txt
     ```
- Inference
    
    > The inference process incldes three steps. First step, we need to predict (segment) the testset. Second step, the segmentation result should be restored into orignal format. Third step, evaluate the model performance using precision, recall, and f-score.
    
    - Predict: 
      > you can find the prediction script in the `train/segmenter` folder, but the corresponding config file for inference in `inference` folder. In this step, you are required to use the converted testset and then segment it using the best model. Besides, you are also required to convert the testset into the same format as [Huang et al. (2020)](https://aclanthology.org/2020.emnlp-main.318.pdf).
      
      ```
      python main_crf.py -c config_test_crf.txt
      ```
    - Restoring:
      > you need to restore the segmented testset into the original formatted text. You can find the corresponding script in `preprocess` folder, as well as it looks highly similar to converting process as we described above.
      
      ```
      python convert_fomat_utils.py -f post -io original_test_file -i segmented_test_file -o final_output_file
      ```
      > The `final_output_file` actually it would be the restored testfile.
      
    - Evaluate: 
      > Evaluation should be done on the restored testset and we use the precision, recll, and f-score as the evaluation method. The corresponding script is also in `preprocess` folder.
      
      ```
      python eval_ch_seg_rslt.py gold_standard_test_file final_output_file
      ```


## Contact
If you have questions, suggestions and bug reports, please email [miradel_51@hotmail.com](miradel_51@hotmail.com); [zheng-yh19@mails.tsinghua.edu.cn](zheng-yh19@mails.tsinghua.edu.cn); [kaiyuhuang@hotmail.com](kaiyuhuang@hotmail.com).
