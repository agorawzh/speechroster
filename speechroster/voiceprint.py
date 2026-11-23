"""Lab260 ReDimNet2+ embeddings and open-set speaker matching."""

import hashlib
import statistics
import sys
from collections import defaultdict
from math import sumprod


SAMPLE_RATE = 16_000
CHECKPOINT_SHA256 = "57056ea3ff1afcf6d825f0f44abc0a4c68fcdd6738ec36e23017df060712f02b"


def decode_audio(path):
    import av
    import numpy as np

    with av.open(str(path)) as media:
        track = next(iter(media.streams.audio), None)
        if track is None:
            raise ValueError(f"No audio track in {path}.")
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=SAMPLE_RATE)
        frames = [block.to_ndarray().reshape(-1) for frame in media.decode(track)
                  for block in resampler.resample(frame)]
        frames.extend(block.to_ndarray().reshape(-1) for block in resampler.resample(None))
    if not frames:
        raise ValueError(f"No audio samples in {path}.")
    return np.concatenate(frames)


class VoiceEncoder:
    def __init__(self, root):
        import torch

        source = root.resolve()
        if not (source / "asv").is_dir():
            source = source / "redimnet2-plus-main"
        weights = source / "weights/pytorch_model_fsdp.bin"
        with weights.open("rb") as file:
            if hashlib.file_digest(file, "sha256").hexdigest() != CHECKPOINT_SHA256:
                raise ValueError(f"Unexpected Lab260 checkpoint contents at {weights}.")
        sys.path.insert(0, str(source))
        from asv.models.redimnet2 import ReDimNet2Encoder

        # The published weights use a 2-D backbone head and a 72-band frontend.
        self.encoder = ReDimNet2Encoder(
            model_name="b6", pretrained=False,
            model_overrides={"out_channels": 224, "return_2d_output": True},
        )
        state = torch.load(weights, weights_only=True, map_location="cpu")
        prefix = "_orig_mod.encoder."
        self.encoder.load_state_dict(
            {key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)},
            strict=True,
        )
        self.encoder.eval()

    def embed(self, samples):
        import torch

        if not 2 * SAMPLE_RATE <= len(samples) <= 60 * SAMPLE_RATE:
            raise ValueError("Use a single-speaker enrollment clip between 2 and 60 seconds.")
        with torch.inference_mode():
            waveform = torch.from_numpy(samples).unsqueeze(0)
            features = self.encoder.spec_frontend(waveform)
            vector = self.encoder(features).squeeze(0)
            norm = vector.norm().item()
            if not 0 < norm < float("inf"):
                raise ValueError("ReDimNet2+ did not return a usable voiceprint.")
            return (vector / norm).tolist()


def voice_windows(segments):
    candidates = defaultdict(list)
    for start, end, label, _ in segments:
        if label == "Unlabeled" or end - start < 2:
            continue
        if any(other_label != label and other_start < end and start < other_end
               for other_start, other_end, other_label, _ in segments):
            continue
        candidates[label].append((start, end))
    return {label: sorted(windows, key=lambda w: w[1] - w[0], reverse=True)[:3]
            for label, windows in candidates.items()}


def identify(vectors, profiles, threshold, margin):
    scores = sorted(
        ((statistics.mean(sumprod(vector, saved) for vector in vectors for saved in enrolled), name)
         for name, enrolled in profiles.items()), reverse=True,
    )
    if not scores:
        return None, None
    score, name = scores[0]
    clear_lead = len(scores) == 1 or score - scores[1][0] >= margin
    return (name if score >= threshold and clear_lead else None), score
