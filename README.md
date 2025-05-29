# Fine-Tuned-GPT-2-for-Text-Generation
A project for generating English text using a fine-tuned GPT-2 model on a custom dataset collected from Kaggle. Includes a Gradio web interface for real-time text generation.

This project uses a dataset collected from Kaggle LinguoGen dataset. The data from 10 different topic files was merged into a single CSV file (`merged_dataset.csv`) for easier training.



##  Features

- Fine-tunes GPT-2 on a custom dataset.
- Generates context-aware English text from any user prompt.
- Evaluation with loss and perplexity.
- Simple Gradio interface for real-time demo.
- Easy to run locally

##  Evaluation Results

- **Loss**: `2.5486`
- **Perplexity**: `12.79`

These results indicate moderate language modeling performance, with reasonable fluency and contextual relevance in generated outputs.

#### Project resources
 - [Kaggle's LinguoGen dataset](https://www.kaggle.com/datasets/jsonali2003/linguogen-text-generation-dataset)
 - [Hugging Face GPT-2 Model](https://huggingface.co/openai-community/gpt2)
 - [Fine-tuning Tutorial by Y. Kenny](https://www.kaggle.com/code/yeeeekenny/model-finetuning/notebook)  
 - [GeeksforGeeks Text Generation Guide](https://www.geeksforgeeks.org/text2text-generations-using-huggingface-model/)

