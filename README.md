# GPT-Custom-Implementation
Local GPT implementation with custom domain-specific data

## Dataset Links

IMDB Sentiment: https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz

Wine mag dataset: winemag-data-130k-v2.json, https://www.kaggle.com/datasets/christopheiv/winemagdata130k

## LLM Landscape Notes

OpenAI: GPT, GPT-2, GPT-3, GPT-3.5-Turbo, GPT-4 (ChatGPT)

Google: LaMDA (prst) (original Bard), PaLM, PaLM 2 (new may_2023, current Bard)

Meta: LLaMA (smaller, more effecient, open)

Stanford: Alpaca (Alpaca 7B, a model fine-tuned from the LLaMA 7B model on 52K instruction-following demonstrations)

## Reading

https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/
https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition/blob/main/notebooks/09_transformer/gpt/gpt.ipynb
https://keras.io/examples/generative/text_generation_with_miniature_gpt/

## GPT Scaling with Model Size (Parameter Number)

The first version of GPT, GPT-1, has 117 million parameters. 

GPT-2, the second version of the Generative Pretrained Transformer developed by OpenAI, has 1.5 billion parameters in its largest version. Smaller versions of GPT-2 were also released, including models with 117M, 345M, and 774M parameters.

GPT-3, developed by OpenAI, has 175 billion parameters. These parameters are the parts of the model that are learned from the training data and then used to generate predictions.

Maybe: GPT-3.5 Turbo likely has around 30 billion parameters

Maybe: GPT-4 is likely to have been trained with 1 trillion or 100 trillion parameters

Kaplan Scaling (Jarad Kaplan)

Chinchilla Scaling (Chinchilla proposed a mathematical formulation, known as compute-optimal scaling laws, to explain how the performance of a language model decreases as the model size (N) and the number of tokens (D) increase.)

(https://www.lesswrong.com/posts/Ea9d9m8eNFWGv6jPq/a-quick-note-on-ai-scaling-asymptotes)

OpenAI researchers found in their empirical studies that the performance of a model scales as a power-law with model size, with an exponent of roughly 0.8. This means that if you double the size of the model (in terms of parameters), the performance (in terms of metrics like perplexity) tends to improve by a factor of about 2^0.8, or 1.74.

Power law scaling laws are a set of empirical observations about how the performance of transformer neural networks in natural language processing (NLP) tasks scales with various aspects of the model and dataset. These observations suggest that certain characteristics of these models follow a power law distribution, a mathematical relationship where a quantity varies as a power of another.

These scaling laws were observed across three primary dimensions:

Model size (number of parameters): Larger models tend to perform better. This scaling often continues well beyond the point where models seem "large enough" — there doesn't seem to be a clear point of diminishing returns. However, there is a trade-off in terms of computational resources, as larger models require more memory and more time to train.

Dataset size: Larger datasets also tend to lead to better performance. This is perhaps less surprising, as more data generally gives the model more information to learn from. However, the scaling law suggests that the relationship between dataset size and performance also follows a power law distribution, with no clear point of diminishing returns.

Training time: Models that are trained for longer tend to perform better, again following a power law distribution. This continues even after the point where a model's performance on a held-out validation set has plateaued, suggesting that additional training is still useful for improving a model's performance on novel data.

These power law scaling laws suggest that, all else being equal, we can often improve the performance of a transformer model by making it bigger, training it on more data, or training it for longer. However, these all come with increased computational costs, and it's an open question how to best balance these trade-offs in practice.

It's important to note that these are empirical observations, based on a range of experiments across various tasks and datasets. While they provide useful heuristics for improving the performance of transformer models, they do not constitute a hard-and-fast rule, and there may well be exceptions or limitations to these scaling laws that are not yet fully understood.

Srivastava, Aarohi, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R. Brown et al. "Beyond the imitation game: Quantifying and extrapolating the capabilities of language models." arXiv preprint arXiv:2206.04615 (2022).

![](img/pic_arduino_nano.png)


## GPT Parameters that can be Tuned

These parameters can be adjusted when interacting with GPT-3:

*model*: The model parameter specifies the version of the model to be used. For example, "text-davinci-002", "text-curie-002", "text-babbage-002", "text-ada-002", etc. Each model has different capabilities in terms of language understanding and generation.

*prompt: This is the input text that you provide to the model. The model will generate a response based on this input. There's no strict limit to the length of the prompt, but keep in mind that the total tokens (input + output) must be under the model's maximum limit (e.g., 4096 tokens for "text-davinci-002").

*max_tokens: This parameter specifies the maximum length of the model’s response. This can be any integer value up to the model's maximum limit. For example, if you set max_tokens to 100, the model's response will be cut off after 100 tokens.

*temperature: This is a value between 0 and 1 that controls the randomness of the model's responses. A higher value (closer to 1) makes the output more random, while a lower value (closer to 0) makes the output more deterministic.

*top_p: Also known as nucleus sampling, this parameter is a value between 0 and 1 that controls the diversity of the response. A higher value will allow more diversity in the responses, whereas a lower value will limit the model's choice of next words to a smaller set.

*frequency_penalty: This is a value between 0 and 1 that controls the penalty for using frequent tokens. A higher value will make the model less likely to use common words and phrases.

*presence_penalty: This is a value between 0 and 1 that controls the penalty for introducing new concepts into the conversation. A higher value will make the model more likely to stick to the topics and concepts already mentioned in the conversation.
