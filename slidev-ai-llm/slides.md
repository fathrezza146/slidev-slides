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

Going back to from Unknown to 1930

<div class="flex justify-center">
  <img src="/old.jpg" alt="Deep Learning" width="512" />
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

<a href="https://huggingface.co/">https://huggingface.co/</a>



---

<h1 class="text-6xl">Simple Code Example</h1>
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

<h1 class="text-6xl">Simple Code Example</h1>
<div class="absolute inset-0 flex items-center justify-center">
```py
def tokenize(examples):
    outputs = tokenizer(examples['text'], truncation=True, padding=True)
    return outputs

tokenized_ds = ds.map(tokenize,  batched=True)

```
</div>

---

<h1 class="text-6xl">Simple Code Example</h1>
<div class="absolute inset-0 flex items-center justify-center">
```py
path = F"/content/gdrive/My Drive/distilbert-dana-review"
training_args = TrainingArguments(num_train_epochs=1,
                                  output_dir=path,
                                  push_to_hub=False,
                                  per_device_train_batch_size=32,
                                  per_device_eval_batch_size=32,
                                  evaluation_strategy="epoch")

data_collator = DataCollatorWithPadding(tokenizer)

trainer = Trainer(model=model, tokenizer=tokenizer,
                  data_collator=data_collator,
                  args=training_args,
                  train_dataset=tokenized_ds["train"],
                  eval_dataset=tokenized_ds["test"],
                  compute_metrics=compute_metrics)

trainer.train()

trainer.save_model()

```
</div>

---

<h1 class="text-6xl">Simple Code Example</h1>
<div class="absolute inset-0 flex items-center justify-center">
```py
from transformers import pipeline, Conversation
import torch

chatbot = pipeline(
            "conversational", 
            model="facebook/blenderbot-400M-distill",
            tokenizer="facebook/blenderbot-400M-distill",
            device=pipe_device)

def handle_message(msg):
    conversation = Conversation(msg)

    # Generate a response using the Hugging Face model
    response = chatbot(conversation)
    reply = response.generated_responses[-1]

    return reply

```
</div>