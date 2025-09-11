from gpt_download import download_and_load_gpt2

# load model
settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")

# check the settings of GPT-2 small (124M)
print("Settings:", settings)
print("Parameters dictionary keys:", params.keys())
print(params["wte"])
print("Token embedding weight tensor dimensions:", params["wte"].shape)