import os
import json
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Import your existing model code
from image_captioning import (
    Vocab, EncoderCNN, DecoderWithAttention
)

# Import metrics
from metrics import CaptionMetrics, calculate_corpus_metrics

# Path to your trained model checkpoint
CHECKPOINT_PATH = "art_caption_model.pth"

# Path to your images folder
IMAGES_FOLDER = "images"

# Path to annotations
ANNOTATIONS_FILE = "annotations.json"

# How many images to evaluate (set to None for all)
MAX_EVAL_SAMPLES = 200  

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(checkpoint_path, device):
    """
    Load trained model from checkpoint.
    
    Returns:
        encoder, decoder, vocab
    """
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruct vocabulary
    vocab = Vocab()
    if 'vocab' in checkpoint:
        vocab.__dict__.update(checkpoint['vocab'])
    else:
        raise KeyError("Checkpoint doesn't contain vocabulary. Check your checkpoint file.")
    
    print(f"Vocabulary loaded: {len(vocab.itos)} words")
    
    # Create models
    encoder = EncoderCNN(encoded_image_size=14, train_cnn=False).to(device)
    decoder = DecoderWithAttention(
        attention_dim=512,
        embed_dim=512,
        decoder_dim=512,
        vocab_size=len(vocab.itos),
        encoder_dim=2048
    ).to(device)
    
    # Load weights
    encoder.load_state_dict(checkpoint['encoder_state'])
    decoder.load_state_dict(checkpoint['decoder_state'])
    
    print("Model weights loaded successfully")
    
    return encoder, decoder, vocab


def generate_caption(encoder, decoder, image, vocab, device, max_len=30):
    """
    Generate caption for a single image.
    
    Args:
        encoder: Trained encoder
        decoder: Trained decoder
        image: PIL Image
        vocab: Vocabulary object
        device: torch device
        max_len: Maximum caption length
        
    Returns:
        Generated caption string
    """
    # Preprocess image (same as training)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    
    image_tensor = transform(image)
    
    # Set to eval mode
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        # Encode image
        enc_out = encoder(image_tensor.unsqueeze(0).to(device))
        
        # Initialize decoder hidden state
        h, c = decoder.init_hidden_state(enc_out)
        
        # Start with <start> token
        word = torch.tensor([vocab.stoi["<start>"]]).to(device)
        caption_words = []
        
        # Generate words one at a time
        for _ in range(max_len):
            # Embed current word
            emb = decoder.embedding(word).squeeze(0)
            
            # Apply attention
            context, _ = decoder.attention(enc_out, h)
            
            # Handle dimension
            if context.dim() == 2:
                context = context.squeeze(0)
            
            # Gating mechanism
            gate = decoder.sigmoid(decoder.f_beta(h))
            context = gate * context
            
            # LSTM step
            lstm_input = torch.cat([emb, context], dim=0).unsqueeze(0)
            h, c = decoder.decode_step(lstm_input, (h, c))
            
            # Predict next word
            out = decoder.fc(h)
            pred = out.argmax(dim=1)
            word = pred
            idx = pred.item()
            
            # Stop if <end> token
            if idx == vocab.stoi["<end>"]:
                break
            
            # Add word to caption
            caption_words.append(vocab.itos.get(idx, "<unk>"))
        
        return " ".join(caption_words)


def evaluate_model(encoder, decoder, vocab, images_folder, annotations, device, max_samples=None):
    """
    Evaluate model on validation set and compute BLEU scores.
    
    Args:
        encoder: Trained encoder
        decoder: Trained decoder
        vocab: Vocabulary
        images_folder: Path to images
        annotations: List of annotation dicts
        device: torch device
        max_samples: Limit evaluation (None for all)
        
    Returns:
        Dictionary with metrics and examples
    """
    print("\n" + "="*70)
    print("STARTING EVALUATION")
    print("="*70)
    
    metrics_calculator = CaptionMetrics()
    
    all_hypotheses = []  # Generated captions
    all_references = []  # Ground truth captions (as lists)
    sample_outputs = []  # Examples to show
    
    # Group annotations by image
    image_to_captions = {}
    for ann in annotations:
        img_name = ann['image']
        if img_name not in image_to_captions:
            image_to_captions[img_name] = []
        image_to_captions[img_name].append(ann['caption'])
    
    # Evaluate
    images_evaluated = 0
    
    print(f"\nEvaluating on {len(image_to_captions)} unique images...")
    if max_samples:
        print(f"   (limited to first {max_samples} for speed)")
    
    for img_name, captions in tqdm(list(image_to_captions.items())[:max_samples], desc="Generating captions"):
        img_path = os.path.join(images_folder, img_name)
        
        # Skip if image doesn't exist
        if not os.path.exists(img_path):
            continue
        
        try:
            # Load image
            image = Image.open(img_path).convert("RGB")
            
            # Generate caption
            generated = generate_caption(encoder, decoder, image, vocab, device)
            
            # Store for metrics
            all_hypotheses.append(generated)
            all_references.append(captions)  # Multiple reference captions
            
            # Save first few examples
            if len(sample_outputs) < 10:
                sample_outputs.append({
                    'image': img_name,
                    'generated': generated,
                    'references': captions
                })
            
            images_evaluated += 1
            
        except Exception as e:
            print(f"\nError processing {img_name}: {e}")
            continue
    
    print(f"\nSuccessfully evaluated {images_evaluated} images")
    
    # Calculate metrics
    print("\nComputing BLEU scores...")
    metrics = calculate_corpus_metrics(all_references, all_hypotheses)
    
    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"\nBLEU Scores:")
    print(f"   BLEU-1: {metrics['BLEU-1']:.4f}")
    print(f"   BLEU-2: {metrics['BLEU-2']:.4f}")
    print(f"   BLEU-3: {metrics['BLEU-3']:.4f}")
    print(f"   BLEU-4: {metrics['BLEU-4']:.4f}  ⭐ (MAIN METRIC)")
    
    print(f"\nSample Outputs:")
    print("-" * 70)
    for i, sample in enumerate(sample_outputs, 1):
        print(f"\nExample {i}: {sample['image']}")
        print(f"  Generated:    {sample['generated']}")
        print(f"  Ground Truth: {sample['references'][0]}")
        if len(sample['references']) > 1:
            print(f"                {sample['references'][1]}")
    
    print("\n" + "="*70)
    
    return {
        'metrics': metrics,
        'samples': sample_outputs,
        'num_evaluated': images_evaluated
    }


def save_results(results, output_file="evaluation_results.json"):
    """Save evaluation results to JSON file."""
    with open(output_file, 'w') as f:
        json.dump({
            'metrics': results['metrics'],
            'samples': results['samples'],
            'num_evaluated': results['num_evaluated']
        }, f, indent=2)
    print(f"\nResults saved to: {output_file}")


def main():
    print("\n" + "="*70)
    print("TRAINED MODEL EVALUATION")
    print("="*70)
    
    # Check files exist
    print("\nChecking files...")
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Checkpoint not found: {CHECKPOINT_PATH}")
        print("\nPlease update CHECKPOINT_PATH in this script to your trained model.")
        return
    
    if not os.path.exists(IMAGES_FOLDER):
        print(f"Images folder not found: {IMAGES_FOLDER}")
        print("\nPlease update IMAGES_FOLDER in this script.")
        return
    
    if not os.path.exists(ANNOTATIONS_FILE):
        print(f"Annotations file not found: {ANNOTATIONS_FILE}")
        print("\nPlease update ANNOTATIONS_FILE in this script.")
        return
    
    print("All files found")
    
    # Load checkpoint
    try:
        encoder, decoder, vocab = load_checkpoint(CHECKPOINT_PATH, DEVICE)
    except Exception as e:
        print(f"\nError loading checkpoint: {e}")
        print("\nMake sure the checkpoint was saved with vocab included.")
        return
    
    # Load annotations
    print(f"\nLoading annotations from: {ANNOTATIONS_FILE}")
    with open(ANNOTATIONS_FILE, 'r') as f:
        annotations = json.load(f)
    print(f"Loaded {len(annotations)} annotations")
    
    # Evaluate
    try:
        results = evaluate_model(
            encoder, decoder, vocab,
            IMAGES_FOLDER, annotations, DEVICE,
            max_samples=MAX_EVAL_SAMPLES
        )
        
        # Save results
        save_results(results)
        
        # Summary
        print("\n" + "="*70)
        print("✅ EVALUATION COMPLETE!")
        print("="*70)
        print(f"\nKey Results:")
        print(f"   BLEU-4 Score: {results['metrics']['BLEU-4']:.4f}")
        print(f"   Images Evaluated: {results['num_evaluated']}")
        print(f"\nNext Steps:")
        print("   1. Check evaluation_results.json for full results")
        print("   2. Build Gradio demo with app.py")
        print("   3. Prepare presentation with these metrics")
        print("="*70)
        
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()