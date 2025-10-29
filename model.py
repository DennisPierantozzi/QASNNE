import math
import torch
import torch.nn as nn

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import ViTModel, BlipTextModel
from peft import get_peft_model, LoraConfig

from transformers import VisualBertConfig
from transformers import VisualBertModel, GPT2Model, ViTModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SurgicalGPTGen(nn.Module):
    def __init__(self, vis_pos_emb=None, peft_config=None):
        super(SurgicalGPTGen, self).__init__()
        self.vis_pos_emb = vis_pos_emb

        # visual feature extractor
        self.img_feature_extractor = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        # visual embed
        VB_config = VisualBertConfig.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
        VB_config.visual_embedding_dim = 768
        visualbert = VisualBertModel(config=VB_config)
        self.visual_embedder = visualbert.embeddings.visual_projection
        # tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # question embedding
        question_embedder = GPT2Model.from_pretrained('gpt2')
        question_embedder.config.pad_token_id = self.tokenizer.eos_token
        self.question_embedder = question_embedder.wte
        # GPT2 decoder
        self.gpt_decoder = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_decoder = get_peft_model(self.gpt_decoder, peft_config)
        self.gpt_decoder.config.pad_token_id = self.tokenizer.eos_token
        self.gpt_decoder.print_trainable_parameters()

    def forward(self, image, qa_inputs):
        if image.dim() == 5:  # 如果有额外的维度 [batch, 1, 3, 224, 224]
            image = torch.squeeze(image, 1)  # 变为 [batch, 3, 224, 224]
        image = image.to(device)

        # get visual features
        img_features = self.img_feature_extractor(image)
        visual_embeds = self.visual_embedder(img_features[0])  # [batch_size, 197, 768]
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float, device=device)
        # get textual features
        qa_input_ids = qa_inputs['input_ids'].to(device)  # [batch_size, text_len]
        qa_att_mask = qa_inputs['attention_mask'].to(device)  # [batch_size, text_len]
        qa_embeds = self.question_embedder(qa_input_ids)  # [batch_size, text_len, 768]
        # concat features: vision first
        inputs_embeds = torch.cat((qa_embeds, visual_embeds), dim=1)
        attention_mask = torch.cat((qa_att_mask.float(), visual_attention_mask), dim=1)

        # decode
        gpt_output = self.gpt_decoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return gpt_output.logits  # [batch_size, 197+text_len, vocab_size]

    def get_tokenizer(self):
        return self.tokenizer


class PitVQAGen(nn.Module):
    def __init__(self, lora_rank=8, lora_alpha=32):
        super().__init__()

        # set peft config
        peft_config = LoraConfig(task_type="CAUSAL_LM",
                                 inference_mode=False,
                                 r=lora_rank,
                                 lora_alpha=lora_alpha,
                                 lora_dropout=0.1)
        
        # visual encoder
        model_name = "google/vit-base-patch16-224-in21k"
        self.visual_encoder = ViTModel.from_pretrained(model_name)

        # tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token  # end of string

        # text encoder
        self.text_encoder = BlipTextModel.from_pretrained("Salesforce/blip-vqa-base")

        # 保存原始预训练的embedding权重
        original_weights = self.text_encoder.embeddings.word_embeddings.weight.data

        # 创建新的embedding层
        new_vocab_size = len(self.tokenizer)
        embedding_dim = self.text_encoder.embeddings.word_embeddings.embedding_dim
        new_embeddings = nn.Embedding(new_vocab_size, embedding_dim)

        # 复制原始权重到新embedding层的对应位置
        original_vocab_size = original_weights.shape[0]
        new_embeddings.weight.data[:original_vocab_size] = original_weights

        # 替换embedding层
        self.text_encoder.embeddings.word_embeddings = new_embeddings

        # gpt decoder with LoRA
        self.gpt_decoder = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_decoder = get_peft_model(self.gpt_decoder, peft_config)

    def forward(self, image, question_inputs):
        # visual encoder
        image = image.to(device)
        image_embeds = self.visual_encoder(image).last_hidden_state  # torch.Size([bs, 197, 768])
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)  # torch.Size([bs, 197])

        question_input_ids = question_inputs['input_ids']  # torch.Size([bs, 25])
        question_att_mask = question_inputs['attention_mask']

        # multimodal encoder
        text_output = self.text_encoder(input_ids=question_input_ids,
                                        attention_mask=question_att_mask,
                                        encoder_hidden_states=image_embeds,
                                        encoder_attention_mask=image_atts,
                                        return_dict=True)
        text_embeds = text_output.last_hidden_state  # torch.Size([bs, 25, 768]), args.question_len=25

        # text decoder
        gpt_output = self.gpt_decoder(inputs_embeds=text_embeds,
                                      encoder_attention_mask=question_att_mask)  # torch.Size([bs, 25, 50257])
        return gpt_output.logits
