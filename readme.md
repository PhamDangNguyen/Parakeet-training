# Parakeet-CTC-1.1B model
## Introduce
Parakeet-ctc-1.1b is an ASR model that transcribes speech in lower case English alphabet. This model is jointly developed by NVIDIA NeMo and Suno.ai teams. It is an XXL version of FastConformer CTC (around 1.1B parameters) model. Parakeet-ctc-1.1b has higher accuracy while the inference time increases insignificantly compared to the fast-conformer model.
## How to train using this repo?
### B1: Set up the environment run
```
sh install.sh
```

followed by the creation of manifest data in the following format:
```
                            .....
{"audio_filepath": "/home/team_voice/STT_pdnguyen/STT_data/STT_dataset/35.augment_08_2023/4041_split_0_time_mask.wav", "text": "có nước ngọt rồi một loại hạt lạ mà ta chưa thấy bao giờ nhưng chim ăn được", "duration": 2.75}

{"audio_filepath": "/home/team_voice/STT_pdnguyen/STT_data/14623_4038_gUTUxyXOu6Q_25_hai_compresison_aac_short_noise.wav", "text": "một giải đấu liệu có thành công trong việc truyền tải hình ảnh đến với các khán giả hay không đó là việc có bao nhiêu khán giả lựa chọn cây chì của mình", "duration": 7.488}
                            .....
```
Note: Default have to 3 fields are "audio_filepath", "text", "duration". You can see an example in 
```
/home/pdnguyen/fast_confomer_finetun/parakeet-training/metadata_train/all_data_val.json
```

### B2: Creating tokenizer following code:
```
sh process_tokenizer.sh
```
With:

--manifest: Information about data that includes audio and text paths.

--data_root: Output tokenizer dir (.model and vocab).

--vocab_size=n: Creating n character or subword.

--tokenizer="spe": Type of tokenizer used. spe is using SentencePiece model.

--no_lower_case: Do not normalize the text to lowercase.

--spe_type="bpe": Type of encoding text method. Here is BPE (Byte Pair Encoding)

### B3:Training model
- Fill into .yaml ("/home/pdnguyen/fast_confomer_finetun/parakeet-training/hparam/parakeet_ctc_bpe.yaml") with fields:
```
train_ds:
    manifest_filepath: ....

validation_ds:
    manifest_filepath: ....

tokenizer:
    dir: ....
    type: ....

trainer:
  devices: 1 # number of GPUs which training used, -1 would use all available GPUs
```
* Training - enter code into the command
```
sh train.sh
```
With:

CUDA_VISIBLE_DEVICES=6 : GPU number 6 which the process will seeing and used. (override 6 by 0,1,2,3,4, ... to using multi GPU for training)

--config-path: Path to config .yaml file.

--config-name: Name of config .yaml file.