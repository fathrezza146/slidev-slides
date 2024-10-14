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

<div class="flex justify-center">
  <img src="/deep.jpg" alt="Deep Learning" width="512" />
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