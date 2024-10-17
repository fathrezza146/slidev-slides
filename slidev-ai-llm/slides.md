---
title: Artificial Intelligence
theme: seriph
class: text-center
transition: slide-bottom
mdc: true
---

# Artificial Intelligence

---
class: text-center
---

<h2>Going back to 1930</h2>

<div class="flex flex-row justify-center">
  <img src="/old.jpg" alt="Deep Learning" width="512" />
  <img src="/alan-turing.jpg" alt="Deep Learning" width="512" />
</div>

---
class: text-center
---

<div class="flex flex-row gap-4 justify-center">
  <img src="/microwave.jpg" alt="Deep Learning" width="512" />
  <img src="/futureandnow.jpg" alt="Deep Learning" width="512" />
</div>

---
class: text-center
---

<h2>Going back to recent years</h2>

  <img class="mx-auto mt-10" src="/geoffrey-hinton.jpg" alt="Deep Learning" width="206" />
  <p>Geoffrey Hinton (Godfather of AI)</p>

---
class: text-center
---

<div class="flex flex-wrap gap-4 justify-center">
<div>
  <img src="/chess-ai.webp" alt="Deep Learning" width="420" />
      <p>CHESS AI</p>
  </div>
  <img src="/microsoft-ai.webp" alt="Deep Learning" width="512"/>
</div>

---
class: text-center
---

<div class="flex flex-wrap gap-4 justify-center p-4">

  <img class="w-96 h-56" src="/kai.jpg" alt="Deep Learning"/>
  <img class="w-96 h-56" src="/stonks.webp" alt="Deep Learning" />
  <img class="w-96 h-56" src="/mobil.jpeg" alt="Deep Learning" />
  <img class="w-96 h-56" src="/Untitled.jpg" alt="Deep Learning"/>

</div>


---
class: text-center
---

<h1>Should we worried?</h1>
<div class="flex flex-row gap-4 justify-center">
  <img src="/losing-job.jpg" alt="Deep Learning" width="512" />
</div>

---
class: text-center
---

<div class="flex justify-center">
  <img src="/deep.jpg" alt="Deep Learning" width="512" />
</div>

---
class: text-center
---

<div class="flex justify-center">
  <img src="/ml-process.png" alt="Deep Learning" width="512" />
</div>

---

<h1>AI Categories</h1>

- 📚 **Natural Language Processing**
  - Text Recognition
  - Text Generation
- 🖼️ **Computer Vision**
  - Image Segmentation
  - Image Classification
  - Object Detection
- 🎵 **Audio Recognition**
  - Speech Recognition
  - Audio Classification

---
class: text-center
---

<div class="absolute inset-0 flex items-center justify-center">
  <div>
    <h1 class="text-6xl">Large Language Model</h1>
    <p class="text-2xl">AI That Understands Human Language that Trained on Very Large Data</p>
  </div>
</div>

---
class: text-center
---

<div class="flex justify-center">
  <img src="/llm.png" alt="Deep Learning" width="512" />
</div>

---
class: text-center
---

<video src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/assisted-generation/gif_2_1080p.mov" autoplay loop></video>

<a href="https://huggingface.co/docs/transformers/en/llm_tutorial">Huggingface LLM Generation</a>

---

<h1 class="text-6xl">Simple Code Example</h1>
<div class="absolute inset-0 flex items-center justify-center">
```py
!pip install datasets
!pip install accelerate -U
!pip install pip install transformers[torch]

# import all functions
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
import numpy as np

from google.colab import drive
drive.mount('/content/drive')
```
</div>



---

<div class="absolute inset-0 flex items-center justify-center">
```py
url = 'https://docs.google.com/spreadsheets/d/1Q0vf3ZXFFKXTQXinkE58YnvC4njhvsk3fnBwrHRW5N8/export?format=csv'

ds = load_dataset('csv',  data_files=url)
ds = ds["train"].train_test_split(test_size=0.2)
ds

model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

```
</div>

---

<div class="absolute inset-0 flex items-center justify-center">
```py
def tokenize(examples):
    outputs = tokenizer(examples['text'], truncation=True, padding=True)
    return outputs

tokenized_ds = ds.map(tokenize,  batched=True)

```
</div>

---

<div class="absolute inset-0 flex items-center justify-center">
```py
path = F"/content/gdrive/My Drive/distilbert-dana-mini"
training_args = TrainingArguments(num_train_epochs=1,
                                  output_dir=path,
                                  push_to_hub=False,
                                  per_device_train_batch_size=32,
                                  per_device_eval_batch_size=32,
                                  learning_rate=5e-5,
                                  evaluation_strategy="epoch")

trainer = Trainer(model=model, tokenizer=model_tokenizer,
                  data_collator=data_collator,
                  args=training_args,
                  train_dataset=tokenized_ds["train"],
                  eval_dataset=tokenized_ds["test"])

trainer.train()

trainer.save_model()

```
</div>

---

<div class="absolute inset-0 flex items-center justify-center">
```py
pipe_kwargs = {
    "top_k": None,
    "batch_size": 16
}

path = F"/content/drive/My Drive/distilbert-dana-review"
text = "Aplikasi Terbaik sepanjang masa"

reward_pipe = pipeline("sentiment-analysis", path, device=-1)
reward_output =  reward_pipe(text, **pipe_kwargs)

print(reward_output)

```
</div>
