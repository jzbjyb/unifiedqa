import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import T5ForConditionalGeneration, BartForConditionalGeneration

class MyBart(BartForConditionalGeneration):
    def forward(self,
                input_ids=None,
                attention_mask=None,
                encoder_outputs=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                decoder_cached_states=None,
                use_cache=False,
                is_training=False,
                return_logprob=False,
                sample_ind=None,
                objective=None,
                num_question=None,
                num_options=None):

        if is_training:
            decoder_start_token_id = self.config.decoder_start_token_id
            _decoder_input_ids = decoder_input_ids.new_zeros(decoder_input_ids.shape)
            _decoder_input_ids[..., 1:] = decoder_input_ids[..., :-1].clone()
            _decoder_input_ids[..., 0] = decoder_start_token_id
        else:
            _decoder_input_ids = decoder_input_ids.clone()

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=_decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
        )
        lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
        logprobs = F.log_softmax(lm_logits, dim=-1)
        logprobs = torch.gather(logprobs, -1, decoder_input_ids.unsqueeze(-1)).squeeze(-1)
        if return_logprob:
            return logprobs.detach()
        if is_training:
            if objective is None:
                loss_fct = nn.CrossEntropyLoss(reduce=False)
                losses = loss_fct(lm_logits.view(-1, self.config.vocab_size),
                                  decoder_input_ids.view(-1))
                loss = torch.sum(losses * decoder_attention_mask.float().view(-1))
            elif objective in {'softmax', 'margin'}:
                logits = (logprobs * decoder_attention_mask.float()).sum(-1)
                if objective == 'softmax':
                    agg = torch.log(logprobs.new_zeros(num_question, num_options))
                    agg = agg.view(-1).scatter_(0, sample_ind, logits).view(num_question, num_options)
                    log_z = torch.logsumexp(agg, -1)
                    log_softmax = agg[:, 0] - log_z
                    loss = -log_softmax.mean()
                else:
                    margin_value = 1.0
                    agg = logprobs.new_zeros(num_question, num_options)
                    agg = agg.view(-1).scatter_(0, sample_ind, logits).view(num_question, num_options)
                    mask = logprobs.new_zeros(num_question, num_options).view(-1).scatter_(0, sample_ind, torch.ones_like(logits)).view(num_question, num_options)
                    margin = torch.clamp(margin_value + (agg[:, 1:] - agg[:, :1]), min=0.0) * mask[:, 1:]
                    loss = margin.sum(-1).mean()
            else:
                raise NotImplementedError
            return loss
        return (lm_logits, ) + outputs[1:]

    def generate_from_string(self, _input, tokenizer=None, **generator_args):
        assert tokenizer is not None
        if isinstance(_input, str):
            _input = [[0] + tokenizer.encode(_input)]
        if isinstance(_input, list) and isinstance(_input[0], str):
            _input = [[0] + tokenizer.encode(i) for i in _input]
        if isinstance(_input, list):
            if isinstance(_input[0], int):
                _input = [_input]
            _input = torch.LongTensor(_input)
        res = self.generate(_input, **generator_args)
        return ([tokenizer.decode(x, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=True).strip() for x in res])

