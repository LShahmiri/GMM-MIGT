"""All Mutual Information modes: auto, grayscale, color - and both colour estimators.

    python examples/02_mi_modes.py [path/to/dataset]
"""

from _demo_data import dataset_from_argv, make_demo_dataset, output_dir

from gmm_migt import GMMMIGTSplitter

dataset = dataset_from_argv()

for mode in ("auto", "grayscale", "color"):
    res = GMMMIGTSplitter(dataset, mode=mode, verbose=False).plan()   # plan() = no files copied
    r = res.classes["leaf"] if "leaf" in res.classes else next(iter(res.classes.values()))
    print(f"mode={mode:<9} class={r.name}: strategy={r.strategy}, bins={[b.size for b in r.bins]}")

# Colour MI estimator: scikit-image NMI (MIGT 1.1.1, default) or
# scikit-learn mutual_info_score per channel (thesis Section 3.3)
res = GMMMIGTSplitter(dataset, mode="color", color_mi="mutual_info_score", verbose=False).plan()
print("color_mi='mutual_info_score':", res.summary().splitlines()[2])

# A grayscale dataset: 'auto' selects grayscale MI automatically
gray = make_demo_dataset(output_dir("gray_dataset"), gray=True)
print(GMMMIGTSplitter(gray, mode="auto", verbose=False).plan().summary())
