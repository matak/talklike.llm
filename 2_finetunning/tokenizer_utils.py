import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def setup_tokenizer_and_model(model_name, base_model):
    """Nastaví tokenizer a model pro fine-tuning"""
    
    # 1. Načtení tokenizeru
    base_tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir='/workspace/.cache/huggingface/transformers',
        local_files_only=False,
        resume_download=True,
        force_download=False
    )
    print(f"📊 Původní délka tokenizeru: {len(base_tokenizer)}")
    
    # 2. Kontrola a přidání pad tokenu
    if base_tokenizer.pad_token is None:
        # Zkusíme použít existující tokeny
        if base_tokenizer.eos_token:
            base_tokenizer.pad_token = base_tokenizer.eos_token
            print(f"✅ Používám EOS token jako PAD: {base_tokenizer.pad_token}")
        else:
            # Přidáme nový pad token
            base_tokenizer.add_special_tokens({"pad_token": "<pad>"})
            print(f"✅ Přidán nový pad token: {base_tokenizer.pad_token}")
            
            # Důležité: Resize model embeddings
            base_model.resize_token_embeddings(len(base_tokenizer))
            print(f"📊 Model embeddings resized na: {len(base_tokenizer)}")
    else:
        print(f"ℹ️ Pad token už existuje: {base_tokenizer.pad_token}")
    
    # 3. Synchronizace s modelem
    if hasattr(base_model.config, 'pad_token_id'):
        old_pad_id = base_model.config.pad_token_id
        base_model.config.pad_token_id = base_tokenizer.pad_token_id
        print(f"🔄 Pad token ID změněn: {old_pad_id} → {base_model.config.pad_token_id}")
    else:
        print("⚠️ Model nemá pad_token_id v config")
    
    # 4. Kontrola konzistence
    try:
        assert base_tokenizer.pad_token_id == base_model.config.pad_token_id, \
            "Tokenizer a model mají různé pad token ID!"
        print(f"✅ Tokenizer a model synchronizovány")
    except AssertionError as e:
        print(f"❌ Chyba synchronizace: {e}")
        # Pokusíme se opravit
        base_model.config.pad_token_id = base_tokenizer.pad_token_id
        print(f"🔧 Opraveno: pad_token_id nastaven na {base_tokenizer.pad_token_id}")
    
    return base_tokenizer, base_model

def check_unknown_tokens(dataset, tokenizer, debugger=None, max_samples_to_check=100):
    """Kontroluje, zda dataset obsahuje neznámé tokeny pro tokenizer"""
    print(f"\n🔍 Kontroluji neznámé tokeny v datasetu...")
    
    unknown_tokens_found = []
    total_unknown_count = 0
    samples_with_unknown = 0
    
    # Kontrolujeme pouze prvních max_samples_to_check vzorků pro rychlost
    samples_to_check = min(max_samples_to_check, len(dataset))
    print(f"📊 Kontroluji {samples_to_check} vzorků z {len(dataset)} celkem")
    
    for i in range(samples_to_check):
        sample = dataset[i]
        input_ids = sample['input_ids']
        
        # Kontrola každého tokenu
        unknown_in_sample = []
        for token_id in input_ids:
            if token_id == tokenizer.unk_token_id:
                unknown_in_sample.append(token_id)
        
        if unknown_in_sample:
            samples_with_unknown += 1
            total_unknown_count += len(unknown_in_sample)
            unknown_tokens_found.append({
                "sample_index": i,
                "unknown_count": len(unknown_in_sample),
                "total_tokens": len(input_ids),
                "unknown_percentage": len(unknown_in_sample) / len(input_ids) * 100
            })
    
    # Výpis výsledků
    print(f"📊 Výsledky kontroly neznámých tokenů:")
    print(f"  Vzorků s neznámými tokeny: {samples_with_unknown}/{samples_to_check}")
    print(f"  Celkový počet neznámých tokenů: {total_unknown_count}")
    print(f"  Procento vzorků s neznámými tokeny: {samples_with_unknown/samples_to_check*100:.1f}%")
    
    if unknown_tokens_found:
        print(f"⚠️ NALEZENY NEZNÁMÉ TOKENY!")
        print(f"📋 Detailní informace o prvních 5 vzorcích s neznámými tokeny:")
        
        for i, info in enumerate(unknown_tokens_found[:5]):
            sample = dataset[info['sample_index']]
            decoded_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=False)
            
            print(f"  Vzorek {info['sample_index']}:")
            print(f"    Neznámých tokenů: {info['unknown_count']}/{info['total_tokens']} ({info['unknown_percentage']:.1f}%)")
            print(f"    Text (prvních 200 znaků): {decoded_text[:200]}...")
            print()
        
        # Uložení detailních informací do debug
        if debugger:
            debugger.save_step("unknown_tokens_check", {
                "samples_checked": samples_to_check,
                "samples_with_unknown": samples_with_unknown,
                "total_unknown_count": total_unknown_count,
                "unknown_percentage": samples_with_unknown/samples_to_check*100,
                "detailed_info": unknown_tokens_found[:10],  # Prvních 10 detailů
                "tokenizer_unk_token_id": tokenizer.unk_token_id,
                "tokenizer_unk_token": tokenizer.unk_token
            }, f"Kontrola neznámých tokenů - nalezeno {samples_with_unknown} vzorků s neznámými tokeny")
        
        # Doporučení pro opravu
        print(f"💡 Doporučení pro opravu:")
        print(f"   1. Zkontrolujte formát dat - možná používáte špatné tagy")
        print(f"   2. Ověřte, že používáte správný tokenizer pro váš model")
        print(f"   3. Zkontrolujte, zda data neobsahují speciální znaky")
        print(f"   4. Pro Mistral/Llama modely zkontrolujte ChatML formát")
        
        # Kontrola, zda pokračovat
        if samples_with_unknown > samples_to_check * 0.5:  # Více než 50% vzorků má neznámé tokeny
            print(f"❌ KRITICKÁ CHYBA: Více než 50% vzorků obsahuje neznámé tokeny!")
            print(f"   Zastavuji fine-tuning. Opravte data před pokračováním.")
            return False
        else:
            print(f"⚠️ VAROVÁNÍ: Některé vzorky obsahují neznámé tokeny, ale pokračuji...")
            return True
    else:
        print(f"✅ Žádné neznámé tokeny nenalezeny!")
        
        if debugger:
            debugger.save_step("unknown_tokens_check", {
                "samples_checked": samples_to_check,
                "samples_with_unknown": 0,
                "total_unknown_count": 0,
                "unknown_percentage": 0.0,
                "status": "OK"
            }, "Kontrola neznámých tokenů - žádné problémy")
        
        return True

def check_tokenizer_compatibility(tokenizer, model_name, debugger=None):
    """Kontroluje kompatibilitu tokenizeru s modelem"""
    print(f"\n🔧 Kontroluji kompatibilitu tokenizeru s modelem...")
    
    # Detekce typu modelu
    is_mistral = "mistral" in model_name.lower()
    is_llama = "llama" in model_name.lower()
    is_dialogpt = "dialogpt" in model_name.lower()
    
    print(f"📊 Model: {model_name}")
    print(f"📊 Typ modelu: {'Mistral' if is_mistral else 'Llama' if is_llama else 'DialoGPT' if is_dialogpt else 'Unknown'}")
    print(f"📊 Vocab size: {len(tokenizer)}")
    print(f"📊 UNK token: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")
    print(f"📊 PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"📊 EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    
    # Kontrola speciálních tokenů
    issues = []
    
    if tokenizer.pad_token is None:
        issues.append("Chybí PAD token")
    
    if tokenizer.eos_token is None:
        issues.append("Chybí EOS token")
    
    # Kontrola očekávaných tokenů podle typu modelu
    if is_mistral or is_llama:
        # Mistral/Llama by měl mít ChatML tokeny
        if not any("[INST]" in tokenizer.decode([i]) for i in range(min(1000, len(tokenizer)))):
            issues.append("Možná chybí ChatML tokeny ([INST], [/INST])")
    elif is_dialogpt:
        # DialoGPT by měl mít speciální tokeny
        if not any("<|" in tokenizer.decode([i]) for i in range(min(1000, len(tokenizer)))):
            issues.append("Možná chybí DialoGPT tokeny (<|system|>, <|user|>, atd.)")
    
    if issues:
        print(f"⚠️ Nalezeny problémy s tokenizerem:")
        for issue in issues:
            print(f"   - {issue}")
        
        if debugger:
            debugger.save_step("tokenizer_compatibility_check", {
                "model_name": model_name,
                "model_type": "Mistral" if is_mistral else "Llama" if is_llama else "DialoGPT" if is_dialogpt else "Unknown",
                "vocab_size": len(tokenizer),
                "unk_token": tokenizer.unk_token,
                "pad_token": tokenizer.pad_token,
                "eos_token": tokenizer.eos_token,
                "issues": issues
            }, f"Kontrola kompatibility tokenizeru - nalezeno {len(issues)} problémů")
        
        return False
    else:
        print(f"✅ Tokenizer je kompatibilní s modelem")
        
        if debugger:
            debugger.save_step("tokenizer_compatibility_check", {
                "model_name": model_name,
                "model_type": "Mistral" if is_mistral else "Llama" if is_llama else "DialoGPT" if is_dialogpt else "Unknown",
                "vocab_size": len(tokenizer),
                "unk_token": tokenizer.unk_token,
                "pad_token": tokenizer.pad_token,
                "eos_token": tokenizer.eos_token,
                "status": "OK"
            }, "Kontrola kompatibility tokenizeru - OK")
        
        return True

def tokenize_function(examples, tokenizer, max_length=2048):
    """Tokenizuje text pro fine-tuning"""
    # Tokenizace s padding pro konzistentní délky
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding=True,  # Povolíme padding
        max_length=max_length,
        return_tensors=None
    )
    
    # Nastavení labels stejné jako input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized 