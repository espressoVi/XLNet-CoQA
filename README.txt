Implementation of Intervention based training for conversational question answering (CoQA dataset) on XLNet.

Features:
1) Train on the original dataset
2) Train on the combined O, TS, TS-R dataset
3) Evaluation on O,TS, TS-R
4) Attention metrics eta, p_sep

Usage:
1) Install the necessary requirements from the environment.yml file by entering

conda env create -f environment.yml
conda activate RobustProject

2) Download the coqa dataset and place in ./data, files can be downloaded from 
https://nlp.stanford.edu/data/coqa/coqa-train-v1.0.json  and
https://nlp.stanford.edu/data/coqa/coqa-dev-v1.0.json
You should have the files 

./data/coqa-dev-v1.0.json
./data/coqa-train-v1.0.json

3) Run main.py with

python main.py --train [O|C] --eval [O|TS|RG] --output [directory]

--train O for original training
--train C for combined training
--eval O for eval on original dataset
--eval TS for eval on TS dataset
--eval RG for eval on TS-R dataset
--output is the ouput directory for saving/loading weights and saving predictions and logs
e.g.
python main.py --train O --eval O --output XLNet_orig (original training and eval on O)
python main.py --eval TS --output XLNet_orig (evaluate the model at XLNet_orig on TS dataset)
python main.py --train C --eval O --output XLNet_orig (combined training and eval on O)

4)Following eval you will have a predict_normal_det.json file at the provided directory. Then run

python ./results/convert_coqa.py --input_file ./[directory]/predict_normal_det.json --output_file pred.json
python evaluate-v1.0.py --data-file data/coqa-dev-v1.0.json --pred-file pred.json

5) Steps 3,4 needs to be repeated for original and combined training and evaluation on all datasets.

6) run attention-qr.py with
python attention-qr.py --output [directory] 

To get eta and p_sep statistics of the model stored at directory. Output is stored in [directory]

7) Repeat step 6 for the original and combined training model.
