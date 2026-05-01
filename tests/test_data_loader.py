import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image

from calibract.data.loaders import load_kaggle_brain_mri


class DataLoaderTests(unittest.TestCase):
    def test_kaggle_loader_accepts_imagefolder_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for class_name in ["glioma", "meningioma", "notumor", "pituitary"]:
                class_dir = root / class_name
                class_dir.mkdir()
                for idx in range(5):
                    image = Image.new("RGB", (32, 32), color=(idx * 20, 40, 80))
                    image.save(class_dir / f"{idx}.jpg")

            train_loader, val_loader, test_loader, num_classes = load_kaggle_brain_mri(
                str(root),
                batch_size=4,
                subset_fraction=1.0,
                seed=7,
            )

            self.assertEqual(num_classes, 4)
            self.assertGreater(len(train_loader.dataset), 0)
            self.assertGreater(len(val_loader.dataset), 0)
            self.assertGreater(len(test_loader.dataset), 0)


if __name__ == "__main__":
    unittest.main()
