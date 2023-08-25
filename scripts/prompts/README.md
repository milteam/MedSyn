# ChatGPT pipeline
To run the script OPENAI_API_KEY environment variable should be set.
To generate a dataset using run:
```bash
python chatgpt_pipeline.py --dir results --samples samples.json --limit 500 --offset 0 --lang ru --gpt gpt-3.5-turbo
```
Where
* dir - the directory where the script stores the resulting JSON files
* sample - the file containing information about diseases used to generate prompts
* limit - the number of samples to generate
* offset -the number of samples to skip in the samples.json
* gpt - gpt model to use

# Generate test set for Alpaca
To generate instructions to test a local model run 
```bash
python generate_test.py -s samples.json -o testset.json
```