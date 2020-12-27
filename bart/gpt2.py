import torch
from torch import nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel

class MyGPT2(GPT2LMHeadModel):
    def forward(self,
                input_ids=None,
                past=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                loss_mask=None,
                use_cache=True,
                sample_ind=None,
                objective=None,
                num_question=None,
                num_options=None):

        transformer_outputs = self.transformer(
            input_ids,
            past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_loss_mask = loss_mask[..., 1:].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            if objective is None:
                # Flatten the tokens
                loss_fct = nn.CrossEntropyLoss(reduce=False)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss = torch.sum(loss * shift_loss_mask.float().view(-1))
            elif objective in {'softmax', 'margin'}:
                logprobs = F.log_softmax(shift_logits, dim=-1)
                logprobs = torch.gather(logprobs, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
                logits = (logprobs * shift_loss_mask.float()).sum(-1)
                if objective == 'softmax':
                    agg = torch.log(logprobs.new_zeros(num_question, num_options))
                    agg = agg.view(-1).scatter_(0, sample_ind, logits).view(num_question, num_options)
                    log_z = torch.logsumexp(agg, -1)
                    log_softmax = agg[:, 0] - log_z
                    loss = -log_softmax.mean()
                elif objective == 'margin':
                    margin_value = 1.0
                    agg = logprobs.new_zeros(num_question, num_options)
                    agg = agg.view(-1).scatter_(0, sample_ind, logits).view(num_question, num_options)
                    mask = logprobs.new_zeros(num_question, num_options).view(-1).scatter_(0, sample_ind, torch.ones_like(logits)).view(num_question, num_options)
                    margin = torch.clamp(margin_value + (agg[:, 1:] - agg[:, :1]), min=0.0) * mask[:, 1:]
                    loss = margin.sum(-1).mean()
            else:
                raise NotImplementedError
            return loss
        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)
