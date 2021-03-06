## Structure
- `cellCNN_clear/data/cellCNN`: contains the scripts for data generation and preprocessing, download the relevant data
  at https://zenodo.org/record/5597098#.YXbaz9ZBzt0 (for NK or AML) or http://flowrepository.org/experiments/2166 (for NIND or RRMS), uncompress and place it in the data/cellCNN/folder
    - gen_data_NK generates centralized and distributed datasets for NK dataset
    - gen_data_AML generates centralized and distributed datasets for AML dataset
    - gen_data_RRMS_NIND generates centralized and distributed datasets for RRMS or NIND datasets

- `cellCNN_clear/eval/`: contains the evaluation plots and the notebook used to generate them

- `cellCNN_clear/layers/`: contains the implementation of neural networks layers

- `cellCNN_clear/protocols/`: contains the implementation of the centralized and distributed protocols. The centralized protocol can be executed with protocols/centralized/centralized_test.go (use flag "-run CellCnn" for the full protocol). The distributed protocol can be executed with protocols/decentralized/cnn_clear_test.go

- `cellCNN_clear/simul/`: contains the simulation configurations (to use with Mininet)

- `cellCNN_clear/utils/`: contains various utilities files
