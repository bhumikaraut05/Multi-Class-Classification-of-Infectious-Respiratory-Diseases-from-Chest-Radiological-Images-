# Project analysis

## 1. Existing project structure
`Chest-X-ray-Classification-main.zip` contains only Readme.md and 7 Jupyter notebooks (no .py files, no weights, no data, no requirements file):
Pre-processing, Baseline CNN, VGG19, ResNet50, Custom-CNN (spatial attention), **Vision Transformer**, Results analysis.

## 2. Existing ML pipeline
5-class chest X-ray classification (COVID-19, Lung-Opacity, Normal, Viral Pneumonia, Tuberculosis) on the Harvard Dataverse set (32,687 images). Data are loaded from `Dataset5_raw_{train,val,test}.h5` (datasets `image`, `image_name`, `image_label`) as 224×224×3 uint8 arrays → `tf.data` → zoom → CLAHE → sharpen → /255 → subtract `mean_image` → batch 128 → one-hot → train with class weights. Test set: 6,540 images.

## 3. Model architecture (final ViT = notebook cells 54–63, `VitB16_DownSized`)
Patch size **20** (not 16 despite the name) → 11×11 = 121 patches (×1200 values) → Dense 768 → class token + learned position embedding (122 tokens) → **5** pre-LN transformer blocks (MultiHeadAttention 5 heads, key_dim 768, ReLU MLP 3072→768, dropout 0.1) → attention pooling (see Known problems) → Dense 2096 → Dropout 0.4 → Dense 2096 → concat with GlobalAveragePooling1D → Dense 5 softmax. **89,710,422 parameters**, which this app reproduces exactly (asserted in `test_prediction.py`). An earlier 16-px ViT (cells 45–52) was superseded and is not used.
Training: AdamW (lr 1e-5, wd 1e-3, warm-up + cosine), categorical cross-entropy, early stopping on val_auc.
Reported test metrics (README): accuracy 0.787, macro-F1 ~0.77, AUC 0.951.

## 4. Input requirements
`(batch, 224, 224, 3)` float32, scaled to [0,1] then minus `mean_image`.

## 5. Preprocessing pipeline (ported verbatim to `backend/app/preprocessing.py`)
1. `zoom_into_chest_area(0.83)` – centre crop, resize back to 224
2. `apply_clahe(grid 15, clip 2.0)` – per channel, uint8
3. `sharpen_image` – kernel `[[0,-1,0],[-1,5,-1],[0,-1,0]]`, `cv2.filter2D`
4. `/255`
5. `− mean_image`

Added at inference (not in notebooks): decode upload → RGB → resize to 224×224 (`INTER_AREA`). **Assumption:** the dataset files were already 224×224 RGB; how they were originally resized is not in the notebooks.

## 6. Class labels
`0 COVID-19, 1 Lung-Opacity, 2 Normal, 3 Viral Pneumonia, 4 Tuberculosis` (notebook `Label_ClassName_Dict`).

## 7. Existing model files
**None.** No `.h5/.keras/.pb` in the ZIP.

## 8. Missing files (needed for real predictions)
- `VitB16_DownSized.h5` – trained weights (loaded by notebook cell 63).
- `mean_image.npy` – never saved by the notebooks (it lived in memory and is deleted in cell 35). Without it the preprocessing cannot match training. See `backend/models/README.md` for export instructions.

## 9. Existing attention visualization
Cells 76–79 build `VisionTransformer_with_attention` using `return_attention_scores=True` per block, then plot. The plotting code slices `[:, 1:, :]` on the *heads* axis and mixes shapes, and the README notes the maps were "inconsistent" (the section is commented out). The model-side extraction is reused; aggregation was rewritten as attention rollout in `attention.py`.

## 10. Required dependencies
TensorFlow ≥2.16 with `tf-keras` (legacy Keras 2, `TF_USE_LEGACY_KERAS=1`), numpy, opencv, Pillow, FastAPI, uvicorn, python-multipart. Original Colab versions were not recorded.

## 11. Recommended architecture
React (Vite) → nginx/Vite proxy → FastAPI → in-memory preprocessing → ViT (loaded once at startup) → JSON incl. base64 PNGs. No database, no stored uploads.

## 12. Known problems
- **"Mean image" quirk (important).** Cell 24 reduces *unbatched* images with `tf.reduce_sum(img, axis=0)`, which sums over image **rows**. The result is a per-column/per-channel mean repeated down all rows, not a true pixel-wise mean. The weights were trained with it, so inference must use the same array. `scripts/compute_mean_image.py` reproduces this behaviour.
- Attention pooling `Dense(1, activation='softmax')` softmaxes over a size-1 axis → always 1.0, i.e. plain sum pooling. Kept as-is for weight compatibility.
- Unused `Embedding` layer in the old `TokenAndPositionEmbedding` (no weights); dropped.
- Readme's results table lists CustomCNN accuracy 0.97 vs its classification report 0.92; worth checking for the thesis.
- Heavy overfitting/under-performance of the ViT (79% test accuracy) – predictions should be treated with caution.

## 13. Steps to run the existing model
Get the two artifacts into `backend/models/`, then `python predict.py xray.png` (from `backend/`).
