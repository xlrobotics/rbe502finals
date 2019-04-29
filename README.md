# rbe502finals
Using seq2seq RNN and GPAR to learn system dynamics of soft snake robot locomotion

This program is mainly inspired by the work in two repositories, and some functionalities are taken directly from the two:
(1) RNN https://github.com/guillaume-chevalier/seq2seq-signal-prediction
(2) GPAR https://github.com/wesselb/gpar

Before running our code, please follow the instructions in (1) and (2) to install all the dependencies, make sure your python version is 3.6

Run seq2seq_gru.py directly to checkout the Seq2seq model trained with real data. The dataset for simulator is too large and thus cannot be upload directly, you can find the real robot data from https://drive.google.com/open?id=17Zuv0K3cMiZaiLPhXORlGqtuXmmKf_43, and simulator data from https://drive.google.com/open?id=1rHqYkN2CHrAj7whvDM0Aj_BAhhz1JySP

Run GPAR_learn.py to checkout the regression model on a single trajectory
