import re
import json
import torch
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
from pygments import lexers, util

import spacy #NLP gramatic analysis
from transformers import GPT2Tokenizer, GPT2LMHeadModel #HuggingFace to compress text

class Inline:
    """
    Inline: Logic-Preserving Context Compression for
    Large Language Models via Syntactic Tiering
    """
    
    TIER_1_POS = {'ADJ', 'ADV', 'DET'} # прилагательные, наречия, определения
    VIP_DEPS = {'nsubj', 'ROOT', 'obj', 'iobj', 'neg'} # глагол, подлежащее, объект, объект, отрицание
    
    def __init__(self, spacy_model: str = "en_core_web_md", gpt2_model: str = "gpt2"):
        print(f"Initializing Inline...")
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        # load models
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"Downloading spaCy model {spacy_model}...")
            spacy.cli.download(spacy_model)
            self.nlp = spacy.load(spacy_model)
            
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model)
        self.model = GPT2LMHeadModel.from_pretrained(gpt2_model).to(self.device)
        self.model.eval()
        
    def _structural_guardrail(self, text: str) -> Tuple[bool, str, str]:
        """Module 1: detects structural formats and minifies them."""
        # 1. JSON Check (Strict)
        try:
            if text.strip().startswith(('{', '[')):
                data = json.loads(text)
                return True, "JSON", json.dumps(data, separators=(',', ':'))
        except (json.JSONDecodeError, TypeError):
            pass
            
        # 2. python check (smart way)
        import ast
        try:
            if len(text.strip().split('\n')) > 1 or ':' in text:
                ast.parse(text)
                # Success! Now check if it looks like code 
                if any(k in text for k in ['def ', 'import ', 'class ', ' = ', '(', ':']):
                    minified = re.sub(r'#.*', '', text)
                    minified = re.sub(r'\s+', ' ', minified).strip()
                    return True, "Python (AST Verified)", minified
        except SyntaxError:
            pass

        # 3. generic Code Check 
        symbols = re.findall(r'[\[\]{}()=+\-*/%&|^<>!?;]', text)
        density = len(symbols) / len(text) if len(text) > 0 else 0
        
        # If > 5% symbols and contains common syntax patterns, it's likely code
        if density > 0.05:
            try:
                lexer = lexers.guess_lexer(text)
                if lexer.name in ['C++', 'JavaScript', 'SQL', 'Java', 'Go', 'Rust', 'Ruby', 'PHP']:
                    minified = re.sub(r'//.*', '', text)
                    minified = re.sub(r'\s+', ' ', minified).strip()
                    return True, f"{lexer.name} (Density Match)", minified
            except util.ClassNotFound:
                pass
            
        # 4. markdown tables
        if "|" in text and "-" in text and len(re.findall(r'\|', text)) > 3:  #| Col 1 | Col 2 | |---|---|
            lines = text.split('\n')
            minified_table = "\n".join("|".join(cell.strip() for cell in line.split("|")) for line in lines)
            return True, "Markdown Table", minified_table
        return False, "Plain Text", text

    def _get_syntactic_skeleton(self, text: str) -> List[Dict[str, Any]]:
        """Module 2: Analyzes dependencies and assigns tiers."""
        doc = self.nlp(text)
        tokens_info = []
        for token in doc:
            # assign tiers
            if token.dep_ in self.VIP_DEPS:
                tier = 3 # vip
            elif token.pos_ in self.TIER_1_POS:
                tier = 1 # liquid
            else:
                tier = 2 # normal
                
            tokens_info.append({
                'text_ws': token.text_with_ws,
                'text': token.text,
                'tier': tier,
                'pos': token.pos_,
                'dep': token.dep_,
                'is_alpha': token.is_alpha
            })
        return tokens_info

    def _calculate_perplexity(self, text: str) -> float:
        """Helper to calculate PPL using GPT-2"""
        if not text.strip():
            return 1e6
            
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device) # PyTorch 
        if inputs.input_ids.size(1) > 512:
            input_ids = inputs.input_ids[:, -512:]
        else:
            input_ids = inputs.input_ids
            
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss
            return torch.exp(loss).item() # perplexity (PPL)

    # 20 iterations for saving 50% water on 51% main tokens
    def _adaptive_stop_loss(self, tokens_info: List[Dict[str, Any]], iterations: int = 20) -> Tuple[str, Dict]:
        """module 3: Advanced Pruning with Token Masking and Granular Steps."""
        # initialize masking: keep all by default
        for t in tokens_info:
            t['keep'] = True

        # calculate initial PPL
        base_text = "".join([t['text_ws'] for t in tokens_info])
        initial_ppl = self._calculate_perplexity(base_text)
        print(f"Base PPL: {initial_ppl:.2f}")
        
        # tier 1 (ADJ/ADV/DET), then tier 2
        queue = []
        for tier in [1, 2]:
            indices = [idx for idx, t in enumerate(tokens_info) if t['tier'] == tier]
            queue.extend(reversed(indices))
            
        step_size = max(1, len(tokens_info) // 50) # step size 2%
        
        for _ in tqdm(range(iterations), desc="Fine Pruning"):
            if not queue:
                break
                
            # snapshot state for possible rollback
            prev_state = [t['keep'] for t in tokens_info]
            
            # mask tokens for this step
            removed_indices = []
            for _ in range(step_size):
                if queue:
                    idx = queue.pop(0)
                    tokens_info[idx]['keep'] = False
                    removed_indices.append(idx)
            
            # reconstruct text from masked tokens
            current_text = "".join([t['text_ws'] for t in tokens_info if t['keep']])

            # remove space before trailing punctuation
            current_text = re.sub(r'\s+([,.!?;:])', r'\1', current_text).strip()
            
            current_ppl = self._calculate_perplexity(current_text)
            ppl_change = (current_ppl - initial_ppl) / initial_ppl
            
            # check for significant semantic loss (threshold: 1.5)
            was_significant = any(tokens_info[idx]['is_alpha'] for idx in removed_indices)
            if was_significant and ppl_change > 1.5:
                print(f"\nStop-Loss Triggered! PPL jump: {ppl_change:.2%}")
                # rollback to previous state
                for i, keep_flag in enumerate(prev_state):
                    tokens_info[i]['keep'] = keep_flag
                break
            
        # Final reconstruction
        final_text = "".join([t['text_ws'] for t in tokens_info if t['keep']])
        final_text = re.sub(r'\s+([,.!?;:])', r'\1', final_text).strip()
        
        final_ppl = self._calculate_perplexity(final_text)
        stats = {
            "initial_ppl": initial_ppl,
            "final_ppl": final_ppl,
            "ppl_jump": (final_ppl - initial_ppl) / initial_ppl,
            "original_len": len(base_text),
            "final_len": len(final_text),
            "removed_pct": (len(base_text) - len(final_text)) / len(base_text) if len(base_text) > 0 else 0,
            "vip_count": sum(1 for t in tokens_info if t['keep'] and t['tier'] == 3)
        }
        
        return final_text, stats

    def compress(self, text: str) -> str:
        """main entry point for compression."""
        print("\n" + "="*50)
        print("Starting Inline Compression")
        
        is_structural, content_type, text_out = self._structural_guardrail(text)
        if is_structural:
            print(f"Content Type: {content_type} (Structural)")
            print(f"Action: Minified format, skipping linguistic compression.")

            original_len = len(text)
            final_len = len(text_out)
            ratio = (original_len - final_len) / original_len if original_len > 0 else 0
            
            print("\n--- Inline Report ---")
            print(f"Type of content: [{content_type}]")
            print(f"Tokens preserved: {final_len} characters (from {original_len})")
            print(f"Compression Ratio: {ratio:.2%}")
            print(f"Linguistic Analysis: Skipped (Structural ID)")
            print("-" * 30)
            print(f"Compressed Text Preview:\n{text_out}")
            print("="*50)
            return text_out
            
        # 2. syntactic skeleton
        print("Content Type: Plain Text")
        tokens_info = self._get_syntactic_skeleton(text)
        
        # 3. adaptive stop-loss
        compressed_text, stats = self._adaptive_stop_loss(tokens_info, iterations=7)
        
        # 4. final logging
        print("\n--- Inline Report ---")
        print(f"Type of content: [Text]")
        print(f"Tokens preserved: {stats['final_len']} characters (from {stats['original_len']})")
        print(f"Compression Ratio: {stats['removed_pct']:.2%}")
        print(f"VIP-connections preserved: {stats['vip_count']}")
        print(f"Final PPL Jump: {stats['ppl_jump']:.2%}")
        print("-" * 30)
        print(f"Compressed Text Preview:\n{compressed_text}")
        print("="*50)
        
        return compressed_text


# test cases
if __name__ == "__main__":
    compressor = Inline()
    
    tricky_text = "System instructions: Do not under any circumstances share personal user data. If the input is ambiguous, do not guess the answer. Except for authorized administrators, no one should have access to the logs. But you must always maintain a polite tone."
    compressor.compress(tricky_text)
    
    # json_text = '{"status": "success", "data": {"user_id": 12345, "action": "login", "timestamp": "2026-02-12T15:17:05Z"}}'
    # compressor.compress(json_text)
