"""
fix_model_compatibility.py
──────────────────────────
Fixes the Keras version mismatch error:
  "Unrecognized keyword arguments: ['batch_shape', 'optional']"

This happens when a model saved with Keras 3.x / TF 2.16+ is loaded
in TF 2.10–2.15 (older Keras 2).

SOLUTION: Rebuild the exact same architecture in YOUR installed Keras,
then transfer weights layer-by-layer from the saved .h5 file.

Run ONCE on your machine — produces  fixed_model.keras  (or .h5).
Then point realtime_ids.py --model fixed_model.keras at it.

Usage:
    py -3.10 fix_model_compatibility.py
    py -3.10 fix_model_compatibility.py --weights bilstm_weights.h5
    py -3.10 fix_model_compatibility.py --out my_fixed.h5
"""

import argparse
import sys

# ── Parse args before importing TF (faster --help) ───────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--model",   default="bilstm_ids.h5",
                help="Original saved model (.h5)")
ap.add_argument("--weights", default=None,
                help="Separate weights file if available (bilstm_weights.h5)")
ap.add_argument("--out",     default="fixed_model.keras",
                help="Output path for the fixed model  [default: fixed_model.keras]")
ap.add_argument("--test",    action="store_true",
                help="Run a quick inference test after saving")
args = ap.parse_args()

import numpy as np

print("Loading TensorFlow …")
import tensorflow as tf
print(f"  TensorFlow : {tf.__version__}")

# ════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Rebuild architecture
#  Must match EXACTLY what you trained:
#    Conv1D(64, 3, relu) → MaxPool(2) → BatchNorm →
#    BiLSTM(128, return_seq=True) → BiLSTM(64) → Dense(64,relu) → Dense(5,softmax)
#
#  If your numbers differ (e.g. 32 filters, 256 LSTM units), edit here.
# ════════════════════════════════════════════════════════════════════════════

INPUT_SHAPE  = (122, 1)   # (time_steps=122, features=1)
NUM_CLASSES  = 5

def build_model():
    inp = tf.keras.Input(shape=INPUT_SHAPE, name="input")

    x = tf.keras.layers.Conv1D(
            filters=64, kernel_size=3, activation="relu",
            padding="same", name="conv1d")(inp)
    x = tf.keras.layers.MaxPooling1D(pool_size=2, name="maxpool")(x)
    x = tf.keras.layers.BatchNormalization(name="batchnorm")(x)

    x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=True),
            name="bilstm_1")(x)
    x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=False),
            name="bilstm_2")(x)

    x = tf.keras.layers.Dense(64, activation="relu",  name="dense_hidden")(x)
    out = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)

    model = tf.keras.Model(inputs=inp, outputs=out, name="bilstm_ids")
    return model

print("\n[1/4] Building fresh model with your architecture …")
new_model = build_model()
new_model.summary()

# ════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Load weights from the saved file
#  Strategy A: load_weights (works even when model_config is incompatible)
#  Strategy B: manual layer-by-layer copy (fallback)
# ════════════════════════════════════════════════════════════════════════════

weight_source = args.weights if args.weights else args.model
print(f"\n[2/4] Loading weights from: {weight_source}")

loaded_ok = False

# ── Strategy A: by_name=True  (robust to layer order differences) ───────────
try:
    new_model.load_weights(weight_source, by_name=True, skip_mismatch=True)
    loaded_ok = True
    print("  ✓ Weights loaded via load_weights(by_name=True)")
except Exception as e:
    print(f"  ✗ Strategy A failed: {e}")

# ── Strategy B: try loading old model with custom_object_scope workaround ───
if not loaded_ok:
    print("  Trying Strategy B: legacy load with compat shim …")
    try:
        # Monkey-patch InputLayer to accept unknown kwargs
        original_init = tf.keras.layers.InputLayer.__init__

        def patched_init(self, *a, **kw):
            kw.pop("batch_shape", None)
            kw.pop("optional", None)
            # batch_shape → input_shape
            if "batch_shape" in kw:
                bs = kw.pop("batch_shape")
                kw["input_shape"] = tuple(bs[1:])
            original_init(self, *a, **kw)

        tf.keras.layers.InputLayer.__init__ = patched_init

        old_model = tf.keras.models.load_model(args.model, compile=False)
        tf.keras.layers.InputLayer.__init__ = original_init  # restore

        # Copy weights layer by layer by matching names
        old_names = {l.name: l for l in old_model.layers}
        new_names = {l.name: l for l in new_model.layers}

        copied, skipped = 0, 0
        for name, new_layer in new_names.items():
            if name in old_names:
                try:
                    new_layer.set_weights(old_names[name].get_weights())
                    copied += 1
                except Exception as ex:
                    print(f"    skip {name}: {ex}")
                    skipped += 1

        print(f"  ✓ Strategy B: copied {copied} layers, skipped {skipped}")
        loaded_ok = True

    except Exception as e2:
        print(f"  ✗ Strategy B failed: {e2}")

# ── Strategy C: try weights.h5 directly if it exists separately ─────────────
if not loaded_ok and args.weights:
    print("  Trying Strategy C: raw weights.h5 …")
    try:
        new_model.load_weights(args.weights)
        loaded_ok = True
        print("  ✓ Strategy C succeeded")
    except Exception as e3:
        print(f"  ✗ Strategy C failed: {e3}")

if not loaded_ok:
    print("\n[WARNING] Could not transfer weights automatically.")
    print("  The architecture is correct but weights are random.")
    print("  You will need to retrain or manually transfer weights.")
    print("  Saving architecture-only model so you can at least test the pipeline.\n")

# ════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Compile & save in format compatible with your Keras version
# ════════════════════════════════════════════════════════════════════════════

new_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print(f"\n[3/4] Saving fixed model → {args.out}")
new_model.save(args.out)
print(f"  ✓ Saved: {args.out}")

# ════════════════════════════════════════════════════════════════════════════
#  STEP 4 — Verify it loads cleanly
# ════════════════════════════════════════════════════════════════════════════

print(f"\n[4/4] Verification: loading {args.out} …")
try:
    verify = tf.keras.models.load_model(args.out)
    print(f"  ✓ Loads cleanly — input shape: {verify.input_shape}")
    print(f"              output shape: {verify.output_shape}")
except Exception as e:
    print(f"  ✗ Load failed: {e}")
    sys.exit(1)

# ── Optional quick inference test ────────────────────────────────────────────
if args.test:
    print("\n[TEST] Running dummy inference (1, 122, 1) …")
    dummy = np.random.rand(1, 122, 1).astype(np.float32)
    preds = verify.predict(dummy, verbose=0)
    labels = ["DOS", "NORMAL", "PROBE", "R2L", "U2R"]
    print(f"  Raw probs : {preds[0].round(4)}")
    print(f"  Prediction: {labels[int(np.argmax(preds))]}")
    print("  ✓ Inference OK")

print(f"""
══════════════════════════════════════════════════════════
  Done!  Fixed model saved to: {args.out}
══════════════════════════════════════════════════════════

Next step — run the IDS with the fixed model:

  py -3.10 realtime_ids.py --iface 5 --model {args.out}

If weight transfer showed warnings, retrain and save with:

  model.save("bilstm_ids.h5")          # same machine, same TF version
  model.save_weights("bilstm_weights.h5")
══════════════════════════════════════════════════════════
""")