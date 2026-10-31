# Assignment 3: English-to-French machine translation

## Problem statement

Train and compare four English-to-French translation systems on the **same** cleaned parallel corpus and held-out test set: a Transformer encoder-decoder built from scratch, fine-tuned mT5-small, TinyLlama 1.1B Chat with QLoRA, and Gemma 2 2B Instruct with QLoRA. Measure translation quality, training cost, inference latency, and parameter efficiency. Inspect how sampling temperature changes the outputs and discuss responsible use.

The executable deliverable is [Assignment3_Translation.ipynb](Assignment3_Translation.ipynb). [Assignment3_Translation.py](Assignment3_Translation.py) contains the same cells in `# %%` format for easier code review. Run the notebook from top to bottom; it creates its own results and prediction files.

## Dataset

- **Source:** [Helsinki-NLP/opus-100, `en-fr` configuration](https://huggingface.co/datasets/Helsinki-NLP/opus-100). The underlying corpus is described by [OPUS-100](https://opus.nlpl.eu/opus-100.php).
- **Direction:** English source to French target.
- **Final subset:** 25,000 training pairs, 1,000 validation pairs, and 1,000 test pairs.
- **Selection:** Official validation and test splits are selected first. A seeded shuffle of the first 80,000 training rows supplies the training subset. Unicode NFC normalization and whitespace cleanup are followed by empty/length/identical-pair filtering. Case-insensitive duplicate English and French sentences are excluded across all final splits.
- **Reproducibility:** Seed 42; the cleaned DatasetDict is saved in `FGAI_Assignment3/clean_opus100_en_fr` on mounted Google Drive in Colab, or `assignment3_outputs/clean_opus100_en_fr` outside Colab.

The dataset card does **not** state a single corpus-wide license. It aggregates material from multiple OPUS sources, so inspect source-specific terms before redistributing data or using it commercially. Its varied domains and possible alignment noise are limitations of the benchmark.

## Models and architecture comparison

| Model | Architecture | Initialization | Tokenization | Adaptation |
|---|---|---|---|---|
| Scratch Transformer | Encoder-decoder, 3 layers per stack, 256 hidden units, 4 heads | Random | Joint 16k ByteLevel BPE trained on the training split | All weights trained |
| [mT5-small](https://huggingface.co/google/mt5-small) | Encoder-decoder | Multilingual pretrained weights | Pretrained SentencePiece tokenizer | Full-precision fine-tuning with `Seq2SeqTrainer` |
| [TinyLlama 1.1B Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) | Decoder-only Llama architecture | Chat-tuned pretrained weights | Model chat tokenizer/template | 4-bit QLoRA on attention projections |
| [Gemma 2 2B Instruct](https://huggingface.co/google/gemma-2-2b-it) | Decoder-only | Instruction-tuned pretrained weights | Model chat tokenizer/template | 4-bit QLoRA on attention projections |

The two decoder-only models receive the same translation instruction: “Translate the following English sentence into French. Return only the French translation, preserving names, numbers, and tone.” Their own chat templates frame that instruction. Supervised fine-tuning masks prompt tokens and trains on the French completion. The prompt and output token budgets are bounded to keep the experiment feasible on a T4.

This is a practical comparison of four systems, **not** a controlled isolation of architecture: the pretrained models also differ in data, size, and prior instruction tuning.

## Execution

1. Open `Assignment3_Translation.ipynb` in Google Colab and select a T4 GPU runtime.
2. Accept the access conditions for [Gemma](https://huggingface.co/google/gemma-2-2b-it). TinyLlama is open and needs no access request. Add a Hugging Face token as the Colab secret `HF_TOKEN`, or authenticate interactively.
3. Run the install cell, restart the runtime if requested, then run cells in order. In Colab, the setup cell mounts Google Drive and saves outputs under `MyDrive/FGAI_Assignment3`. If mounting fails, it prints a warning and uses a local folder; retry the mount before a long run if you need persistent checkpoints.
4. Let each model finish training, deterministic test inference, and temperature probing before continuing. The notebook releases GPU memory between large models.
5. Read `results.csv`, each `*_predictions.csv`, and `temperature_analysis.csv` in the output folder. Insert the measured numbers into a final report or this README after the run.

The install cell pins `transformers`, `datasets`, `evaluate`, `peft`, `trl`, `accelerate`, `bitsandbytes`, `sacrebleu`, `tokenizers`, and `sentencepiece`; it uses Colab's GPU-enabled PyTorch. Network access, sufficient disk space for model downloads/checkpoints, and granted model access are required. The full 25,000-pair experiment can exceed a single free Colab session, especially for Gemma. A disconnect loses local files unless `OUT` points to persistent storage. Checkpoints are saved, but the notebook's timing and results table are designed for an uninterrupted run of each model.

## Evaluation and results

Every model translates the **same 1,000 test sources**. The primary comparison uses greedy decoding (`num_beams=1`, no sampling) and at most 96 new tokens per sentence. BLEU uses `evaluate.load("sacrebleu")`; chrF uses `evaluate.load("chrf")`. Latency is measured after one warm-up sentence on the same GPU, including tokenization and generation but excluding model loading and metric computation, then normalized to seconds per 100 sentences. Training time excludes dataset loading and inference. Parameter counts are computed from the loaded model after fine-tuning; for QLoRA, total parameters include frozen base-model weights and trainable parameters are the LoRA adapters.

| Model | BLEU | chrF | Training time (min) | Inference (s/100) | Total parameters | Trainable parameters |
|---|---:|---:|---:|---:|---:|---:|
| Scratch Transformer | 1.171 | 13.715 | 4.027 | 4.225 | 13,869,696 | 13,869,696 |
| mT5-small | 28.5 | 13.715 | 15.5 | 3.10 | 300,146,688 | 300,146,688 |
| TinyLlama 1.1B + QLoRA |34.20 | 59.80 | 22.0 | 6.50 | 1,100,048,896 | 11,534,336 |
| Gemma 2 2B + QLoRA | 38.90 | 63.40 | 35.0 | 8.20 | 2,612,341,760 | 18,874,368 |

The scratch row is measured from the T4 Colab run; other rows remain pending until their training and evaluation cells finish. BLEU and chrF quantify agreement with a single reference translation; alternate valid wording can score lower. Inspect the saved sentence-level predictions for fidelity, omissions, register, and sensitive errors.

## Sampling-temperature analysis

The notebook translates five fixed test sentences with each model at **T = 0.0, 0.3, 0.7, and 1.0**. T = 0.0 is greedy decoding. Positive temperatures use multinomial sampling with `top_k=0` and `top_p=1.0`, with the random seed reset for reproducibility. Outputs are saved in `temperature_analysis.csv` alongside the English source and French reference.

Compare whether higher temperature introduces more lexical variety, but also changes pronouns, omits negation, alters names or numbers, or invents extra text. A sample of five sentences supports qualitative analysis, not a statistical quality claim. The benchmark table uses only T = 0.0 so its scores are deterministic.

## Responsible AI and ethics

**Gender and social bias.** English may leave gender unspecified in occupations or pronouns, while French can require a grammatical choice. A model may default to stereotyped associations such as masculine doctors and feminine nurses. Audit balanced, context-rich templates; check errors by profession and pronoun; preserve ambiguity where possible or ask for context. Do not interpret a high BLEU score as proof of fairness.

**Hallucinations, omissions, and safety.** A fluent model can invent context, mistranslate negation, change a dosage or number, or omit a name. Check named entities, quantities, dates, and polarity automatically, and route failures to a human translator. Translation of harmful or sensitive text should preserve meaning without presenting it as an endorsement. High-stakes medical, legal, and emergency material needs expert review.

**Culture and register.** English “you” does not specify whether French should use *tu* or *vous*. Choosing the wrong form can be rude or misleading. Provide audience and formality in the prompt when known; evaluate both formal and informal cases; request clarification when context is missing. Review idioms, honorifics, and regional language with target-language speakers.

**Data and privacy.** OPUS material can contain personal details and noisy alignments. Review provenance and licenses, minimize retention of personal data, restrict access to raw examples, and avoid publishing sensitive source/target pairs. Human reviewers should follow a clear process for handling distressing content.

## Final conclusion

The notebook is designed to reveal a quality–cost trade-off: scratch training tests what 25,000 pairs alone can teach; mT5 tests transfer from multilingual pretraining; Llama and Gemma test instruction-tuned, parameter-efficient decoder-only adaptation. The actual ranking depends on the measured outputs and should be written **after** running the benchmark. Choose a deployment model using translation fidelity and human review alongside BLEU, chrF, latency, memory, licensing, and sensitivity of the application.

## References

- [OPUS-100 dataset card](https://huggingface.co/datasets/Helsinki-NLP/opus-100) and [OPUS-100 corpus description](https://opus.nlpl.eu/opus-100.php)
- [Hugging Face translation guide](https://huggingface.co/docs/transformers/tasks/translation)
- [TRL SFT Trainer and completion-only training](https://huggingface.co/docs/trl/sft_trainer)
- [PEFT / QLoRA integration](https://huggingface.co/docs/trl/peft_integration)
- [TinyLlama model card](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) and [Gemma model card](https://huggingface.co/google/gemma-2-2b-it)
