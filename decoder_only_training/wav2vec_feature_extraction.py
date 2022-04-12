import torch
from pathlib import Path
from transformers.utils.logging import get_logger

logger = get_logger(__name__)

def extract_features_to_files(model, feature_extractor, dataset_split, batch_size, output_path, max_len, sampling_rate):
    Path(output_path).mkdir(exist_ok=True, parents=True)

    for i in range(0, len(dataset_split), batch_size):
        batch = dataset_split[i:i+batch_size]
        audio_batch, sentence_batch = batch["audio"], batch["sentence"]
        for eg in audio_batch:
            if len(eg["array"]) > max_len:
                logger.warning(f"Truncating example of length {len(eg['array'])} to {max_len}.")
                eg["array"] = eg["array"][:max_len]
        features = feature_extractor([eg["array"] for eg in audio_batch],
                                     sampling_rate=sampling_rate,
                                     return_tensors="pt",
                                     padding="longest")

        with torch.no_grad():
            out = model(features.input_values.to(model.device), attention_mask=features.attention_mask.to(model.device))

        assert len(sentence_batch) == len(audio_batch) == len(out.last_hidden_state)
        for sent, audio, hs in zip(sentence_batch, audio_batch, out.last_hidden_state.bfloat16().cpu()):
            file_name = f"{Path(audio['path']).stem}.pt"
            torch.save(
                # .clone() is necessary: https://github.com/pytorch/pytorch/issues/1995
                {"sentence": sent, "wave2vec_features": hs.clone()},
                output_path / file_name
            )