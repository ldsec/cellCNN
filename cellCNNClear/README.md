## Structure
- `cellCNN_clear/data/`: contains the scripts for data generation and preprocessing
    - gen_data_NK generates centralized and distributed datasets for NK dataset [1]. Download the relevant data
at https://imsb.ethz.ch/research/claassen/Software/cellcnn.html, uncompress and place it in the data/cellCNN/ folder
    - gen_data_AML generates centralized and distributed datasets for AML dataset[1]. Download the relevant data
at https://imsb.ethz.ch/research/claassen/Software/cellcnn.html, uncompress and place it in the data/cellCNN/ folder
    - gen_data_RRMS_NIND generates centralized and distributed datasets for RRMS and NIND datasets[2]. Download the relevant data
at  http://flowrepository.org/experiments/2166/, uncompress and place 'discovery cohort' in the data/cellCNN/FlowRepository folder
    - Other files and folders include the relevant metadata for each of the aforementioned dataset.


[1] E. Arvaniti and M. Claassen. Sensitive detection of rare disease-associated cell subsets via representation learning.Nat Commun, 8:1–10, 2017
[2] Galli, E. et al. GM-CSF and CXCR4 define a t helper cell signature in multiple sclerosis. Nat. medicine 25, 1290 – 1300
(2019)

- `cellCNN_clear/eval/`: contains the evaluation plots and the notebook used to generate them for centralized tests

- `cellCNN_clear/layers/`: contains the implementation of neural networks layers

- `cellCNN_clear/protocols/`: contains the implementation of the centralized and distributed protocols
    The centralized protocol can be executed with protocols/centralized/centralized_test.go
        (use flag "-run CellCnn" for the full protocol)
    The distributed protocol can be executed with protocols/decentralized/cnn_clear_test.go

- `cellCNN_clear/simul/`: contains the simulation configurations (to use with Mininet)

- `cellCNN_clear/utils/`: contains various utilities files
