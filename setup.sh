
# method (1): install them one by one
# # install torch, pyg
# conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
# conda install pyg -c pyg

# # install gensim to use word2vec
# pip install gensim==3.8.1
# conda install pandas
# pip install cpgclientlib

# method (2): install from conda
conda env create -n devign -f env.yaml
conda activate devign


# unzip pre-built cpg graph from joern
tar -xzvf cpg.tar.gz

# use word2vec to emb the token
python main.py -e

# train an eval
python main.py -pS