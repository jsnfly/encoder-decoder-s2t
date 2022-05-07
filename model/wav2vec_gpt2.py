import torch
from transformers import SpeechEncoderDecoderModel
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.models.encoder_decoder.modeling_encoder_decoder import shift_tokens_right
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.utils import logging

logger = logging.get_logger(__name__)

class Wav2VecGPT2Model(SpeechEncoderDecoderModel):
    """
    Basically the same as `SpeechEncoderDecoderModel` but position embeddings (initialized with GPT2's position
    embeddings) are added to encoder output
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_encoder_output = self.decoder.config.n_positions
        self.encoder_outputs_pos_emb = nn.Embedding(self.n_encoder_output, self.decoder.config.n_embd)
        with torch.no_grad():
            self.encoder_outputs_pos_emb.weight.copy_(self.decoder.transformer.wpe.weight)
        self.enc_to_dec_proj_ln = nn.LayerNorm(self.decoder.config.n_embd, eps=self.decoder.config.layer_norm_epsilon)

    def forward(
        self,
        inputs=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        input_values=None,
        input_features=None,
        return_dict=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if encoder_outputs is None:
            if inputs is None:
                if input_values is not None and input_features is not None:
                    raise ValueError("You cannot specify both input_values and input_features at the same time")
                elif input_values is not None:
                    inputs = input_values
                elif input_features is not None:
                    inputs = input_features
                else:
                    raise ValueError("You have to specify either input_values or input_features")

            encoder_outputs = self.encoder(
                inputs,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)

        encoder_hidden_states = encoder_outputs[0]

        # optionally project encoder_hidden_states
        if (
            self.encoder_output_dim != self.decoder.config.hidden_size
            and self.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        # Add the position embeddings (initialized with GPT2's position embeddings) to the encoder outputs.
        if encoder_hidden_states.shape[1] > self.n_encoder_output:
            logger.warning(
                f"The provided inputs lead to an encoder output length of {encoder_hidden_states.shape[1]}. "
                f"The added position embeddings can only handle a lenght up to {self.n_encoder_output}. "
                f"Truncating the encoder outputs to {self.n_encoder_output}."
            )
            encoder_hidden_states = encoder_hidden_states[:, :self.n_encoder_output, :]

        encoder_hidden_states += self.encoder_outputs_pos_emb(
            torch.arange(0, encoder_hidden_states.shape[1], device=encoder_hidden_states.device)
        )
        encoder_hidden_states = self.enc_to_dec_proj_ln(encoder_hidden_states)

        # compute correct encoder attention mask
        if attention_mask is not None:
            encoder_attention_mask = self.encoder._get_feature_vector_attention_mask(
                encoder_hidden_states.shape[1], attention_mask
            )
        else:
            encoder_attention_mask = None

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        # Compute loss independent from decoder (as some shift the logits inside them)
        loss = None
        if labels is not None:
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.view(-1))

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_hidden_states,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )