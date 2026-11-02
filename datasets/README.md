# Datasets

The demo and tests use real, small public datasets. Only subsets are committed so the repository
and APK stay small.

| File | Source | Subset | License |
| --- | --- | --- | --- |
| `iris.csv` | UCI Iris / scikit-learn package data | 150 rows, 4 features, 3 classes | public domain / CC BY 4.0 (UCI) |
| `mnist_train_images.idx`, `mnist_train_labels.idx` | MNIST (LeCun et al.) via the PyTorch mirror | 2,000 images | CC BY-SA 3.0 |
| `mnist_test_images.idx`, `mnist_test_labels.idx` | MNIST | 500 images | CC BY-SA 3.0 |
| `sms.tsv` | UCI SMS Spam Collection | 1,200 messages (161 spam) | CC BY 4.0 (UCI) |

MNIST is a subset of the official IDX files. The loader normalizes pixels to `[0,1]` and yields
`[N,1,28,28]`; labels are `INT32`.

SMS labels are `0=ham`, `1=spam`; text is lowercased, tokenized with `[a-z0-9']+`, truncated to
24 tokens, and mapped to a 1,200-word vocabulary built from the subset.

Iris is standardized per feature. The demo uses an 80/20 held-out split; MNIST uses the official
train/test split subset.

## Reproduction

The committed files are sufficient to run the demo. To regenerate the MNIST subset, download the
four official IDX gzip files, parse the headers, and write the first 2,000 train / 500 test images
back as IDX (`2051`) and labels (`2049`). The SMS subset is the first 1,200 rows after a
seed-42 shuffle of `SMSSpamCollection`.
