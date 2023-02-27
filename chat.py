import pytorch_lightning as pl

MODEL_NAME ='t5-base'

class BioQAModel(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)


  def forward(self, input_ids, attention_mask, labels=None):
    output = self.model(
        input_ids, 
        attention_mask=attention_mask,
        labels=labels)

    return output.loss, output.logits

  def training_step(self, batch, batch_idx):
    input_ids = batch['input_ids']
    attention_mask=batch['attention_mask']
    labels = batch['labels']
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("train_loss", loss, prog_bar=True, logger=True)
    return {"loss": loss, "predictions":outputs, "labels": labels}

  def validation_step(self, batch, batch_idx):
    input_ids = batch['input_ids']
    attention_mask=batch['attention_mask']
    labels = batch['labels']
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("val_loss", loss, prog_bar=True, logger=True)
    return loss

  def test_step(self, batch, batch_idx):
    input_ids = batch['input_ids']
    attention_mask=batch['attention_mask']
    labels = batch['labels']
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("test_loss", loss, prog_bar=True, logger=True)
    return loss

  def configure_optimizers(self):

    optimizer = AdamW(self.parameters(), lr=0.0001)
    return optimizer


#model.load_state_dict(checkpoint['model_state_dict'])
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

'''
optimizer - AdamW
T5 Conditional Generator in which we'll give conditions
T5 tokenizer because it is fast
training the model without a learning rate
'''

import torch
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

#best-checkpoint.ckpt

model_directory = '/Users/beeoladeji/Downloads/best-checkpoint.ckpt'

checkpoint = torch.load('/Users/beeoladeji/Downloads/best-checkpoint.ckpt', map_location=torch.device('cpu'))

#get model
model = BioQAModel.load_from_checkpoint(model_directory)

# save the model in HF format
model.model.save_pretrained("hf_model")

# load the saved hf_model using
trained_model = T5ForConditionalGeneration.from_pretrained("hf_model")

tokenizer = T5Tokenizer.from_pretrained('t5-base')

#model = T5ForConditionalGeneration.from_pretrained(model_directory, return_dict=False)
# trained_model = BioQAModel.load_from_checkpoint("checkpoints/best-checkpoint.ckpt")
# trained_model.freeze() #

def generate_answer2(question):
  source_encoding=tokenizer(
      # question["question"],
      # question['Context'],
      question,
      #[input],
      max_length = 396,
      padding="max_length",
      # truncation="only_second",
      truncation="only_first",
      return_attention_mask=True,
      add_special_tokens=True,
      return_tensors="pt"

  )

  generated_ids = trained_model.generate(
      input_ids=source_encoding["input_ids"],
      attention_mask=source_encoding["attention_mask"],
      num_beams=1,  # greedy search
      max_length=80,
      repetition_penalty=2.5,
      early_stopping=True,
      use_cache=True)
  
  preds = [
          tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
          for generated_id in generated_ids
  ]

  return "".join(preds)

if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = generate_answer2(sentence)
        print(resp)

