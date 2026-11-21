# Experiments

The runs behind rlvitals' evaluation. They are not part of the package: they need Python 3.11+,
a GPU, and `trl==1.18.2` on top of `rlvitals[transformers]`.

## The pilot (pilot.py)

One healthy GRPO run: Qwen2.5-0.5B-Instruct on GSM8K's training problems, 200 steps of 4
prompts x 8 completions, learning rate 1e-6 (constant), KL coefficient 0.04, completions up to
256 tokens, float16 autocast on a T4. It measures the step time and what TRL logs, so the
fault runs can be priced and the pass thresholds proposed.

On Kaggle (a phone-verified account; Settings: Accelerator GPU T4 x2, Internet on), in a new
notebook, one cell each:

    !git clone https://github.com/Argo-Navis-Lab/rlvitals
    %cd rlvitals
    !git checkout COMMIT
    !pip install -q trl==1.18.2 -e ".[transformers]"
    !CUDA_VISIBLE_DEVICES=0 python experiments/pilot.py --out /kaggle/working/pilot
    !rlvitals replay /kaggle/working/pilot/metrics.jsonl --trl

COMMIT is the full hash of the commit to run. `CUDA_VISIBLE_DEVICES=0` keeps the run on one
T4: with two visible, the Trainer splits every batch across both. Run it with Save Version ->
Save & Run All (Commit), which keeps going with the browser closed; /kaggle/working/pilot is
then in the version's output.

On a Mac or any CPU, a tiny model checks that the script runs (about a minute):

    uv run --no-project --python 3.12 --with trl==1.18.2 --with-editable ".[transformers]" python experiments/pilot.py --model trl-internal-testing/tiny-Qwen2ForCausalLM-2.5 --out /tmp/pilot-try --steps 4 --problems 16 --completion-tokens 32
