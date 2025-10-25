from collections import Counter
import math


class CaptionMetrics:
    """
    Calculate evaluation metrics for generated captions.
    
    Usage:
        metrics = CaptionMetrics()
        references = ["a dog is playing in the park"]  # ground truth
        hypothesis = "a dog playing in a park"  # model generated
        bleu4 = metrics.bleu_score(references, hypothesis, n=4)
    """
    
    def __init__(self):
        pass
    
    def tokenize(self, text):
        """
        Simple tokenization (same as used in training).
        Converts to lowercase and splits on whitespace.
        """
        return text.lower().strip().replace('.', '').replace(',', '').split()
    
    def bleu_score(self, references, hypothesis, n=4, weights=None):
        """
        Calculate BLEU score (Bilingual Evaluation Understudy).
        
        BLEU measures how many n-grams in the generated caption match the reference captions.
        Higher scores (0-1) mean better match.
        
        Args:
            references: List of reference captions (ground truth)
            hypothesis: Generated caption (model output)
            n: Maximum n-gram order to consider (default 4 for BLEU-4)
            weights: Custom weights for each n-gram order (default: uniform)
        
        Returns:
            BLEU score between 0 and 1
            
        Example:
            references = ["a dog runs in grass", "dog running on grass"]
            hypothesis = "a dog running in the grass"
            # BLEU-4 will check 1-grams, 2-grams, 3-grams, 4-grams overlap
        """
        # Tokenize inputs
        ref_tokens_list = [self.tokenize(ref) for ref in references]
        hyp_tokens = self.tokenize(hypothesis)
        
        # Default weights: uniform distribution
        if weights is None:
            weights = [1.0/n] * n
        
        # Calculate precision for each n-gram order
        precisions = []
        for i in range(1, n+1):
            p_i = self._modified_precision(ref_tokens_list, hyp_tokens, i)
            precisions.append(p_i)
        
        # Calculate brevity penalty to discourage very short captions
        bp = self._brevity_penalty(ref_tokens_list, hyp_tokens)
        
        # BLEU is geometric mean of precisions, multiplied by brevity penalty
        if min(precisions) > 0:
            log_precisions = [w * math.log(p) for w, p in zip(weights, precisions)]
            bleu = bp * math.exp(sum(log_precisions))
        else:
            bleu = 0.0
        
        return bleu
    
    def _modified_precision(self, references, hypothesis, n):
        """
        Calculate modified n-gram precision.
        
        This counts how many n-grams from hypothesis appear in references,
        with "clipping" to avoid rewarding repetition.
        
        Args:
            references: List of tokenized reference captions
            hypothesis: Tokenized hypothesis caption
            n: N-gram order (1=unigram, 2=bigram, etc.)
        """
        # Get n-grams from hypothesis
        hyp_ngrams = self._get_ngrams(hypothesis, n)
        
        if not hyp_ngrams:
            return 0.0
        
        # Count n-grams, but "clip" to max count seen in any reference
        # This prevents rewarding repeated words
        total_count = 0
        clipped_count = 0
        
        for ngram in hyp_ngrams:
            # Count how many times this n-gram appears in hypothesis
            hyp_count = hyp_ngrams[ngram]
            
            # Count max times this n-gram appears in any reference
            max_ref_count = 0
            for ref in references:
                ref_ngrams = self._get_ngrams(ref, n)
                max_ref_count = max(max_ref_count, ref_ngrams.get(ngram, 0))
            
            # "Clip" the count to prevent over-rewarding repetition
            clipped_count += min(hyp_count, max_ref_count)
            total_count += hyp_count
        
        # Precision = clipped matches / total n-grams in hypothesis
        return clipped_count / total_count if total_count > 0 else 0.0
    
    def _get_ngrams(self, tokens, n):
        """
        Extract n-grams from a list of tokens.
        
        Example:
            tokens = ["a", "dog", "runs"]
            n = 2
            returns: Counter({("a", "dog"): 1, ("dog", "runs"): 1})
        """
        ngrams = Counter()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams[ngram] += 1
        return ngrams
    
    def _brevity_penalty(self, references, hypothesis):
        """
        Calculate brevity penalty to discourage overly short captions.
        
        If hypothesis is shorter than references, apply a penalty.
        This prevents "gaming" the metric by generating very short captions.
        """
        hyp_len = len(hypothesis)
        
        # Find closest reference length
        ref_lens = [len(ref) for ref in references]
        closest_ref_len = min(ref_lens, key=lambda x: abs(x - hyp_len))
        
        if hyp_len >= closest_ref_len:
            return 1.0
        else:
            return math.exp(1 - closest_ref_len / hyp_len) if hyp_len > 0 else 0.0
    
    def bleu_1(self, references, hypothesis):
        """BLEU-1: Only considers unigram (single word) matches."""
        return self.bleu_score(references, hypothesis, n=1)
    
    def bleu_2(self, references, hypothesis):
        """BLEU-2: Considers unigram and bigram matches."""
        return self.bleu_score(references, hypothesis, n=2)
    
    def bleu_3(self, references, hypothesis):
        """BLEU-3: Considers up to trigram matches."""
        return self.bleu_score(references, hypothesis, n=3)
    
    def bleu_4(self, references, hypothesis):
        """BLEU-4: Standard metric, considers up to 4-grams (most commonly used)."""
        return self.bleu_score(references, hypothesis, n=4)
    
    def compute_all_metrics(self, references, hypothesis):
        """
        Compute all BLEU metrics at once.
        
        Returns:
            Dictionary with BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
        """
        return {
            'BLEU-1': self.bleu_1(references, hypothesis),
            'BLEU-2': self.bleu_2(references, hypothesis),
            'BLEU-3': self.bleu_3(references, hypothesis),
            'BLEU-4': self.bleu_4(references, hypothesis),
        }


def calculate_corpus_metrics(all_references, all_hypotheses):
    """
    Calculate average metrics across entire dataset (corpus-level evaluation).
    
    Args:
        all_references: List of lists, where each inner list contains reference captions for one image
        all_hypotheses: List of generated captions, one per image
    
    Returns:
        Dictionary with averaged metrics
        
    Example:
        all_references = [
            ["a cat sits", "cat sitting"],  # Image 1 references
            ["a dog runs", "dog running"]   # Image 2 references
        ]
        all_hypotheses = [
            "a cat is sitting",  # Image 1 generated
            "a dog is running"   # Image 2 generated
        ]
    """
    metrics_calculator = CaptionMetrics()
    
    # Accumulate scores
    bleu1_scores = []
    bleu2_scores = []
    bleu3_scores = []
    bleu4_scores = []
    
    for refs, hyp in zip(all_references, all_hypotheses):
        scores = metrics_calculator.compute_all_metrics(refs, hyp)
        bleu1_scores.append(scores['BLEU-1'])
        bleu2_scores.append(scores['BLEU-2'])
        bleu3_scores.append(scores['BLEU-3'])
        bleu4_scores.append(scores['BLEU-4'])
    
    # Return averages
    return {
        'BLEU-1': sum(bleu1_scores) / len(bleu1_scores) if bleu1_scores else 0.0,
        'BLEU-2': sum(bleu2_scores) / len(bleu2_scores) if bleu2_scores else 0.0,
        'BLEU-3': sum(bleu3_scores) / len(bleu3_scores) if bleu3_scores else 0.0,
        'BLEU-4': sum(bleu4_scores) / len(bleu4_scores) if bleu4_scores else 0.0,
    }


# Example usage and testing
if __name__ == "__main__":
    # Test the metrics
    metrics = CaptionMetrics()
    
    # Example 1: Perfect match
    refs = ["a dog is running in the park"]
    hyp = "a dog is running in the park"
    print("Perfect match:")
    print(metrics.compute_all_metrics(refs, hyp))
    print()
    
    # Example 2: Partial match
    refs = ["a dog is running in the park"]
    hyp = "a dog runs in a park"
    print("Partial match:")
    print(metrics.compute_all_metrics(refs, hyp))
    print()
    
    # Example 3: Multiple references (more realistic)
    refs = [
        "a dog is running in the park",
        "dog running through park",
        "a dog plays in a park"
    ]
    hyp = "a dog is playing in the park"
    print("Multiple references:")
    print(metrics.compute_all_metrics(refs, hyp))
    print()
    
    # Example 4: Poor match
    refs = ["a dog is running in the park"]
    hyp = "cat sleeping on couch"
    print("Poor match:")
    print(metrics.compute_all_metrics(refs, hyp))