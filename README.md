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

- **SIGHAN08**
    - It contains "CTB and SXU" corpora. We take CTB6 as CTB dataset in our whole experiment
    - The related [paper](https://aclanthology.org/I08-4010.pdf) and [datasets](https://github.com/hankcs/multi-criteria-cws/tree/master/data/other/) 

- **SIGHAN10**
    - It contains data in different domains, and we choose "Finance, Literature and Medicine" for our cross-domain experiment.
    - The related [paper](https://aclanthology.org/W10-4126.pdf)
    - Please look into `dataset` folder.

- **OTHER**
    - It contains "CNC, UDC and ZX" corpora.
    - The related [paper](https://aclanthology.org/E14-1062.pdf) and [datasets](https://github.com/hankcs/multi-criteria-cws/tree/master/data/other/) 
    

## Requirements
- Preprocessing
    - We use the same data pre-processing as used in [Huang et al. (2020)](https://aclanthology.org/2020.emnlp-main.318.pdf)
    - We use the original format of AS and CITYU instead of using their corresponding simplified versions.
    - Other related scripts are given in `preprocess` folder. 
- Environment
    - Python 3.6
    - torch>=1.4.0
    - transformers>=4.4.2

## Usage
- Preprocessing
    - Cleaning data: we clean the data by using simple three steps instead of dirctly train the openly available corpus.
        
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
        python shuffle_corpus.py rm_dp_file
        ```
        
    - Generating nosiy data: The noisy data generation consists of two steps. 
    
        > First, convert the sequence into char sequence. Second, generate "BMES" 4 classes randomly.
    
        step1, run `con_char.py` to achieve the char sequence; 
        ```
        python con_char.py original_file char_file
        ```
    
        step2, run `gen_4class_rnd.py` to generate the randomly segmented files based on char sequence that achieved from 'step1'; 
        ```
        python gen_4class_rnd.py char_file random_labeled_file 
        ```
     - Cutting the corpus according to the punctuation
        ```
        python cut.py input_file output_file
        ```
     - Convert both the training file and test file into the same format as EMNLP2020. The digits are converted into '0', and the alphabets are converted into 'X'.

        > Convert the training file:
        ```
        python convert_fomat_utils.py -f p-train -i input_file -o output_file
        ```

        > Convert the test file:
        ```
        python convert_fomat_utils.py -f p-test -i input_file -o output_file
        ```
        
- Training ...
    - PTM
    - Revised MRT
- Inference ...


## Contact
If you have questions, suggestions and bug reports, please email [miradel_51@hotmail.com](miradel_51@hotmail.com).
