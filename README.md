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
- Preprocessing ...
    - Nosiy data:
- Dependencies
- Training ...
    - PTM
    - Revised MRT
- Inference ...


## Contact
If you have questions, suggestions and bug reports, please email [miradel_51@hotmail.com](miradel_51@hotmail.com).
