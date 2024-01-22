# gpt_the_office

A gpt that can learn writing about texts given a dataset. The motivation behind this repo is to create scripts from `The Office`. The first iteration will be utilizing character-wise tokens. Inspiration came from Andrej Karpathy and his video [here](https://www.youtube.com/watch?v=kCc8FmEb1nY)

## Setting up the environment
Assumption is you have python3, using pip, and a linux os.
1. `gh repo clone shassen14/gpt_the_office` (or however you would like to clone the repo)
2. `cd gpt_the_office`
3. `sh set_environment.sh` or `bash set_environment.sh`

If one is not running linux, then do something similar to the following:
1. `gh repo clone shassen14/gpt_the_office` (or however you would like to clone the repo)
2. `cd gpt_the_office`
3. `pip3 install virtualenv` (if you don't already have virtualenv installed)
4. `python3 -m venv ./venv` to create your new environment (called 'venv' here)
5. `source venv/bin/activate` to enter the virtual environment
6. `pip3 install -r requirements.txt` to install the requirements in the current environment

## Obtaining the dataset
Assumption is you are already in gpt_the_office directory.
1. `python3 ./data/prep_char_data.py`

The output should look something similar to the following:
```
length of dataset in characters: 21,680
all the unique characters: 
 !(),-.:;?ABCDEFGHIJKLMNOPRSTUVWXYabcdefghijklmnopqrstuvwxyz–‘’“”…
vocab size: 67
train has 19,512 tokens
val has 2,168 tokens
```

This will rather download the dataset or confirm it's already there. One can edit the `data/data_config.py` to edit file names and download another dataset recommended in the comments or one's own.

## Running the program


## Visuals


## Future Plans

- [ ] Collect different level of tokens via tiktoken (what gpt2 uses)
- [x] Create a gpt model based off of [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)
- [x] Save parameters
- [ ] Organize parameters
- [ ] Organize gpt model directory. might make it one file?
- [x] Have a train.py for training and a generate.py for running models given parameters
- [ ] Need to rename main.py to something else
- [ ] Save parameters midway. Have the ability to resume training if stopped midway
- [ ] need better visualization
- [ ] simplify environment setup
- [ ] might include docker as well for environment setup?