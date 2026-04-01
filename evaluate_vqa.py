"""Evaluate edited images using LangSAM segmentation + BLIP-2 VQA scoring."""

import argparse, glob, os, csv
import numpy as np
import torch
from PIL import Image
from lang_sam import LangSAM
from transformers import Blip2ForConditionalGeneration, Blip2Processor


def crop_to_mask(image, mask, padding=10):
    """Crop image to bounding box of mask with padding."""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return image  # no mask found, return full image
    x0, x1 = max(0, xs.min() - padding), min(image.width, xs.max() + padding)
    y0, y1 = max(0, ys.min() - padding), min(image.height, ys.max() + padding)
    return image.crop((x0, y0, x1, y1))


def ask_vqa(processor, model, image, question, device):
    """Ask a VQA question and return the answer string."""
    inputs = processor(images=image, text=question, return_tensors="pt").to(device, torch.float16)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=30)
    return processor.decode(output[0], skip_special_tokens=True).strip().lower()


def score_image(processor, model, image, questions, device):
    """Score an image by asking multiple VQA questions. Returns per-question answers."""
    results = {}
    for q in questions:
        answer = ask_vqa(processor, model, image, q, device)
        results[q] = answer
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate edited images with LangSAM + BLIP-2 VQA")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing edited_XXXX.png files")
    parser.add_argument("--object", type=str, required=True, help="Object to segment with LangSAM (e.g. 'panda')")
    parser.add_argument("--questions", type=str, nargs="+", default=None, help="Custom VQA questions (optional)")
    parser.add_argument("--expected_answer", type=str, default=None, help="Expected answer for 'what is this' question (e.g. 'panda')")
    parser.add_argument("--include_refs", action="store_true", help="Also score ref_*_edited.png files")
    parser.add_argument("--blip2_model", type=str, default="Salesforce/blip2-flan-t5-xl", help="BLIP-2 model name")
    args = parser.parse_args()

    # Default questions if not provided
    if args.questions is None:
        obj = args.object
        args.questions = [
            f"What animal is in this image?",
            f"Does this {obj} look realistic? Answer yes or no.",
            f"Is this a high quality photo? Answer yes or no.",
            f"Are there any visual artifacts or distortions? Answer yes or no.",
        ]

    expected = args.expected_answer or args.object

    # Collect image paths
    edited = sorted(glob.glob(os.path.join(args.image_dir, "edited_*.png")))
    if args.include_refs:
        edited += sorted(glob.glob(os.path.join(args.image_dir, "ref_*_edited.png")))

    if not edited:
        print(f"No edited images found in {args.image_dir}")
        return

    print(f"Found {len(edited)} images to score")
    print(f"Object for segmentation: {args.object}")
    print(f"Expected answer: {expected}")
    print(f"Questions: {args.questions}")
    print()

    # Load models
    print("Loading LangSAM...")
    langsam = LangSAM()

    print("Loading BLIP-2...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Blip2Processor.from_pretrained(args.blip2_model)
    model = Blip2ForConditionalGeneration.from_pretrained(args.blip2_model, torch_dtype=torch.float16).to(device)
    print("Models loaded.\n")

    # Score all images
    all_results = []
    for path in edited:
        name = os.path.basename(path)
        img = Image.open(path).convert("RGB")

        # Segment object with LangSAM
        results = langsam.predict([img], [args.object])
        masks = results[0]["masks"]

        if len(masks) > 0:
            mask = masks[0]
            cropped = crop_to_mask(img, mask)
            segmented = True
        else:
            cropped = img
            segmented = False

        # Ask VQA questions on cropped region
        answers = score_image(processor, model, cropped, args.questions, device)

        # Simple scoring: check if the "what is this" answer matches expected
        first_answer = list(answers.values())[0]
        identity_match = expected.lower() in first_answer

        print(f"  {name} (segmented={segmented}):")
        for q, a in answers.items():
            print(f"    Q: {q}")
            print(f"    A: {a}")
        print(f"    -> Identity match: {identity_match}")
        print()

        row = {"filename": name, "segmented": segmented, "identity_match": identity_match}
        row.update({f"q{i}": a for i, a in enumerate(answers.values())})
        all_results.append(row)

    # Summary
    match_count = sum(1 for r in all_results if r["identity_match"])
    seg_count = sum(1 for r in all_results if r["segmented"])
    print(f"{'='*50}")
    print(f"Total: {len(all_results)} | Segmented: {seg_count} | Identity match: {match_count}/{len(all_results)}")
    print(f"{'='*50}")

    # Save CSV
    csv_path = os.path.join(args.image_dir, "vqa_scores.csv")
    if all_results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
