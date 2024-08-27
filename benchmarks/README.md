# Benchmarking 

## Downloading the ShareGPT dataset

You can download the dataset by inputting under current directory:s
```bash
wget -P ../data/ https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

To run the experiments, we have to download models and specify the models and path we need:

```bash
python dowonload_models.py
```

