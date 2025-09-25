import os
import json
import time
import math
import logging
import pickle
from typing import Dict, List, Tuple, Optional
import torch
import requests

logger = logging.getLogger(__name__)

# =========================
# LLM enrichment (Appendix A)
# =========================

# 给LLM的系统提示，强制约束输出格式，必须返回两个独立的JSON对象
# 必须返回两个独立的JSON对象，并且包含所有原始关系名称作为键，不允许添加额外解释
SYSTEM_INSTRUCTION = (
    "Provide exactly two separate JSON objects in your response, corresponding to each step, strictly in the order presented above. "
    "Do not include additional explanations or metadata beyond the specified JSON objects. "
    "Always provide both JSON objects and ensure they contain all the original relation names as keys."
)

# 构建LLM的提示词函数，根据关系示例构建提示词
def build_prompt(rel_to_example: Dict[str, Tuple[str, str, str]]) -> str:
    # rel_to_example:建为原始关系名，值为知识图三元组
    # Build prompt exactly as described in Appendix A
    header = (
        "LLM Prompt for Relation Text Enrichment\n"
        "You will be provided with a list of relation names, each accompanied by exactly one example triple\n"
        "from a knowledge graph. Follow the instructions below carefully, strictly adhering to the output\n"
        "formats specified.\n"
        "Step 1: Convert Relation Names to Human-Readable Form.\n"
        "Clean each provided relation name, converting it into plaintext, human-readable form.\n"
        "Output Format (JSON Dictionary): {{\n"
        "\"original_relation_name1\": \"Clean Human-Readable Form\",\n"
        "\"original_relation_name2\": \"Clean Human-Readable Form\",\n"
        "...\n"
        "}}\n"
        "Step 2: Generate Short Descriptions\n"
        "For each provided relation, generate a concise description (3-4 words) that clearly captures its\n"
        "semantic meaning based on the given example triple as context. Also, for each relation, generate a\n"
        "description of its supposed inverse relation. These descriptions will be converted into embeddings\n"
        "using jinaai/jina-embeddings-v3 to uniquely identify relations and to measure semantic similarities.\n"
        "So, avoid using common or generic words excessively, and do NOT reuse other relation names, to\n"
        "prevent false semantic similarities. Follow the rules below,\n"
        "Be Concise and Precise: Use as few words as possible while clearly conveying the core meaning.\n"
        "Avoid filler words, unnecessary adjectives, and overly generic language.\n"
        "Emphasize Key Semantics: Focus on the distinctive action or relationship the relation name implies.\n"
        "Ensure that the description highlights the unique aspects that differentiate it from similar relations.\n"
        "Handle Negation Carefully: If the relation involves negation (e.g., “is not part of”), state the\n"
        "negation explicitly and unambiguously. Ensure that the description for a negated relation is clearly\n"
        "distinguishable from its affirmative counterpart.\n"
        "Avoid Common Stopwords as Filler: Do not use common stopwords or phrases that add little\n"
        "semantic content. Every word should contribute meaning. Do not use repetitive words to avoid\n"
        "creating false semantic similarities.\n"
        "Take care of symmetry: Ensure that for relations that are symmetric, the description does not change\n"
        "for its inverse relation.\n"
        "Output Format (JSON Dictionary): {{\n"
        "\"original_relation_name1\": [\"concise description\", \"concise inverse relation description\"],\n"
        "\"original_relation_name2\": [\"concise description\", \"concise inverse relation description\"],\n"
        "...\n"
        "}}\n"
        "Step 3: Merge the Two JSONs\n"
        "Combine the two JSON objects from the previous steps. The final output format should be a direct\n"
        "concatenation of the results from the first two steps: two JSON dictionaries, one after the other,\n"
        "with no additional text, separators, or wrappers.\n"
        "Example (exactly two JSON objects back-to-back):\n"
        "{\n"
        "  \"original_relation_name1\": \"Clean Human-Readable Form\",\n"
        "  \"original_relation_name2\": \"Clean Human-Readable Form\",\n"
        "  ...\n"
        "}\n"
        "{\n"
        "  \"original_relation_name1\": [\"concise description\", \"concise inverse relation description\"],\n"
        "  \"original_relation_name2\": [\"concise description\", \"concise inverse relation description\"],\n"
        "  ...\n"
        "}\n"
        "List of Relations:\n"
    )
    # 构建提示词的主体部分，将关系示例转换为提示词
    body_lines = []
    for r, (h, rr, t) in rel_to_example.items():
        body_lines.append(f"relation_name: \"{r}\" ; example: (\"{h}\", \"{rr}\", \"{t}\")")
    return header + "\n".join(body_lines)

# 封装DMX API的调用逻辑，向GPT-4o发送提示词并获取响应
# prompt: build_prompt生成的LLM提示词
# api_key：身份认证密钥
# model:模型名称
def _call_dmx_api(prompt: str, api_key: str, model: str = "gpt-4o-2024-11-20") -> str:
    """
    Call DMX API for GPT-4o model
    """
    url = "https://www.dmxapi.cn/v1/chat/completions"
    headers = {
        "Accept": "application/json",
        "Authorization": api_key,
        "User-Agent": "DMXAPI/1.0.0",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"DMX API call failed: {e}")

# LLM增强关系语义函数  
def enrich_relations_with_llm(
    rel_to_example: Dict[str, Tuple[str, str, str]],
    api_key: str,
    model: str = "gpt-4o-2024-11-20",
    max_retries: int = 3,
) -> Tuple[Dict[str, str], Dict[str, Tuple[str, str]]]:
    """
    Returns:
        cleaned: {rel: cleaned_name} 清洁后的关系名
        descs: {rel: (desc, inverse_desc)} 关系描述
    """
    # 构建LLM的提示词
    prompt = build_prompt(rel_to_example)
    last_err = None
    
    for i in range(max_retries):
        # 调用DMX API，发送提示词并获取响应
        try:
            # 打印调用次数
            print(f"Calling DMX API (attempt {i+1}/{max_retries})...")
            content = _call_dmx_api(prompt, api_key, model)
            content = content.replace('json', '').replace('```', '').strip()
            # 解析两个JSON对象
            # Parse two JSON objects back-to-back
            objs = [] # 存储两个JSON对象
            buf = "" # 存储JSON对象的字符串
            depth = 0 # 存储JSON对象的深度
            for ch in content:
                buf += ch
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            objs.append(json.loads(buf))
                            buf = ""
                        except Exception:
                            pass
            print(objs)
            # 如果返回的JSON对象数量小于2，则抛出错误
            if len(objs) < 2:
                raise ValueError("LLM did not return two JSON objects")
            # 解析两个JSON对象
            cleaned_relations = objs[0]
            relation_descriptions = objs[1]
            
            # normalize 标准化
            cleaned = {k: str(v) for k, v in cleaned_relations.items()}
            descs = {k: (v[0], v[1]) for k, v in relation_descriptions.items()}
            
            print(f"Successfully processed {len(cleaned)} relations")
            return cleaned, descs
            
        except Exception as e:
            # 记录错误
            last_err = e
            print(f"Attempt {i+1} failed: {e}")
            # 如果重试次数小于最大重试次数，则等待2秒后重试
            if i < max_retries - 1:
                time.sleep(2.0 * (i + 1))  # Exponential backoff
    # 如果重试次数大于最大重试次数，则抛出错误
    raise RuntimeError(f"LLM enrichment failed after {max_retries} attempts: {last_err}")

# 优先从缓存中加载关系语义，如果缓存不存在，则通过LLM生成关系语义
def load_or_generate_relation_semantics(
    rel_to_example: Dict[str, Tuple[str, str, str]],
    cache_file: str,
    api_key: str,
    model: str = "gpt-4o-2024-11-20",
    force_regenerate: bool = False,
) -> Tuple[Dict[str, str], Dict[str, Tuple[str, str]]]:
    """
    Load relation semantics from cache or generate new ones via LLM
    """
    # 如果不需要强制重新生成，并且缓存文件存在，则从缓存中加载关系语义
    if not force_regenerate and os.path.exists(cache_file):
        try:
            print(f"Loading relation semantics from cache: {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Verify cache contains required keys
            if 'cleaned' in cached_data and 'descs' in cached_data:
                cleaned = cached_data['cleaned']
                descs = cached_data['descs']
                print(f"Loaded {len(cleaned)} relations from cache")
                return cleaned, descs
            else:
                print("Cache file format invalid, regenerating...")
        except Exception as e:
            print(f"Failed to load cache: {e}, regenerating...")
    
    # Generate new semantics
    print("Generating new relation semantics via LLM...")
    cleaned, descs = enrich_relations_with_llm(rel_to_example, api_key, model)
    
    # 保存到缓存
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'cleaned': cleaned,
                'descs': descs,
                'rel_names': list(rel_to_example.keys()),
                'timestamp': time.time()
            }, f)
        print(f"Saved relation semantics to cache: {cache_file}")
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")
    
    return cleaned, descs


# =========================
# Embeddings with jina-embeddings-v3
# =========================
# 使用jina-embeddings-v3对关系的文本信息（关系名、清洁后的关系名、关系描述）生成文本嵌入
def embed_with_jina_hf(texts: List[str], model_name: str = "jinaai/jina-embeddings-v3") -> torch.Tensor:
    # texts: 关系的文本信息（关系名、清洁后的关系名、关系描述）
    """
    Use jina-embeddings-v3 model downloaded from HuggingFace Hub
    Automatically downloads the model on first use
    """
    try:
        from transformers import AutoModel
        import torch
    except Exception as e:
         raise RuntimeError("transformers and torch are required. Please install: pip install transformers torch") from e
    
    # Download model from HuggingFace Hub (will cache locally after first download)
    print(f"Loading jina-embeddings-v3 from HuggingFace: {model_name}")
    model = AutoModel.from_pretrained(
        model_name, 
        trust_remote_code=True, # 信任远程代码
        cache_dir=None,  # 使用默认缓存目录
        force_download=False,  # 使用缓存版本
        resume_download=True   # 继续下载
    )
    
    # When calling the `encode` function, you can choose a `task` based on the use case:
    # 'retrieval.query', 'retrieval.passage', 'separation', 'classification', 'text-matching'
    # For relation text matching in knowledge graphs, we use 'text-matching'
    print(f"Generating embeddings for {len(texts)} texts using jina-embeddings-v3...")
    embeddings = model.encode(texts, task="text-matching")
    
    print(f"Generated embeddings with shape: {embeddings.shape}")
    
    # Return as torch tensor with float32 dtype
    return torch.tensor(embeddings, dtype=torch.float)

# 组合关系嵌入
def build_relation_embeddings(
    rel_names: List[str],
    cleaned: Dict[str, str],
    descs: Dict[str, Tuple[str, str]],
    combine: str = "COMBINED_SUM",
) -> torch.Tensor:
    """
    combine in {REL_NAME, LLM_REL_NAME, LLM_REL_DESC, COMBINED_SUM, COMBINED_AVG}
    Returns (R, d)
    """
    rel_texts = [] # 关系名
    cleaned_texts = [] # 清洁后的关系名
    desc_texts = [] # 关系描述
    
    for r in rel_names: # 遍历关系名
        rel_texts.append(r)
        cleaned_texts.append(cleaned.get(r, r))
        # use only forward description for embedding; inverse used separately if needed
        desc, inv_desc = descs.get(r, (r, r)) # 获取关系描述
        desc_texts.append(desc) # 添加关系描述
    
    # Generate embeddings using jina-embeddings-v3
    E_rel = embed_with_jina_hf(rel_texts) # 生成关系名嵌入
    E_clean = embed_with_jina_hf(cleaned_texts) # 生成清洁后的关系名嵌入
    E_desc = embed_with_jina_hf(desc_texts) # 生成关系描述嵌入
    
    if combine == "REL_NAME": # 返回关系名嵌入
        return E_rel
    if combine == "LLM_REL_NAME": # 返回清洁后的关系名嵌入
        return E_clean
    if combine == "LLM_REL_DESC": # 返回关系描述嵌入
        return E_desc
    if combine == "COMBINED_SUM": # 返回关系名嵌入、清洁后的关系名嵌入和关系描述嵌入的和
        return E_rel + E_clean + E_desc
    if combine == "COMBINED_AVG": # 返回关系名嵌入、清洁后的关系名嵌入和关系描述嵌入的平均
        return (E_rel + E_clean + E_desc) / 3.0
    
    raise ValueError("Unknown combine policy")

# 生成反向关系嵌入（取负法）
def make_inverse_by_negation(E: torch.Tensor) -> torch.Tensor:
    """
    τ_{r^{-1}} = -τ_r (inverse relation embedding by negation)
    """
    return torch.cat([E, -E], dim=0)


# =========================
# Text Relation Graph Construction
# =========================
# 文本关系图构建
def build_text_relation_graph(graph, rel_text_init, threshold=0.8, top_percent=None):
    """
    Build textual relation graph G_R^{TEXT} based on cosine similarity
    """
    '''
        graph:原始知识图对象
        rel_text_init:关系嵌入矩阵
    '''
    import torch.nn.functional as F
    
    # 对嵌入矩阵进行L2归一化
    rel_text_norm = F.normalize(rel_text_init, p=2, dim=1)
    
    # 计算相似度矩阵
    sim_matrix = torch.mm(rel_text_norm, rel_text_norm.t())
    
    # 保留前k%相似度的边
    if top_percent is not None:
        # Top-k per row
        k = max(1, int(top_percent * rel_text_init.shape[0]))
        _, top_indices = torch.topk(sim_matrix, k=k, dim=1)
        mask = torch.zeros_like(sim_matrix)
        mask.scatter_(1, top_indices, 1)
        sim_matrix = sim_matrix * mask
    else: # 保留超过阈值的边
        # Threshold-based
        mask = sim_matrix >= threshold
        sim_matrix = sim_matrix * mask.float()
    
    # 去除自环
    sim_matrix.fill_diagonal_(0)
    
    # 提取边索引和权重
    edge_index = torch.nonzero(sim_matrix > 0, as_tuple=False).t()
    edge_weight = sim_matrix[edge_index[0], edge_index[1]]
    
    # 创建文本关系图 存储图的核心信息
    text_rel_graph = torch.utils.data.Data()
    text_rel_graph.edge_index = edge_index
    text_rel_graph.edge_weight = edge_weight
    text_rel_graph.num_nodes = rel_text_init.shape[0]
    
    # Attach to original graph
    graph.text_relation_graph = text_rel_graph
    graph.rel_text_init = rel_text_init
    
    print(f"Built text relation graph with {edge_index.shape[1]} edges from {rel_text_init.shape[0]} relations")
    
    return graph


# =========================
# Test Functions
# =========================

# 测试
def test_jina_embeddings():
    """
    Test function to verify jina-embeddings-v3 from HuggingFace is working correctly
    """
    test_texts = [
        "Follow the white rabbit.",  # English
        "Sigue al conejo blanco.",  # Spanish
        "Suis le lapin blanc.",  # French
        "跟着白兔走。",  # Chinese
        "اتبع الأرنب الأبيض.",  # Arabic
        "Folge dem weißen Kaninchen.",  # German
    ]
    try:
        print("Testing jina-embeddings-v3 downloaded from HuggingFace...")
        embeddings = embed_with_jina_hf(test_texts)
        print(f"✅ Successfully generated embeddings with shape: {embeddings.shape}")
        
        # Compute similarities (like in the example)
        similarity = embeddings[0] @ embeddings[1].T
        print(f"📊 Similarity between first two texts: {similarity.item():.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Error testing jina embeddings: {e}")
        print("Please ensure:")
        print("  1. transformers is installed: pip install transformers")
        print("  2. Internet connection is available for downloading from HuggingFace")
        print("  3. Sufficient disk space for model cache (~1.5GB)")
        return False


def test_dmx_api():
    """
    Test DMX API connection
    """
    api_key = "sk-dvJb2Dz7h6ikAnEAXP7morrMIIxTcpQN6EPKTUza7AqmSkNc"
    
    test_prompt = "Hello, are you working?"
    try:
        response = _call_dmx_api(test_prompt, api_key)
        print(f"✅ DMX API test successful: {response[:100]}...")
        return True
    except Exception as e:
        print(f"❌ DMX API test failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing SEMMA text semantics module...")
    print("=" * 60)
    
    # Test jina embeddings
    print("\n1. Testing jina-embeddings-v3...")
    jina_success = test_jina_embeddings()
    
    # Test DMX API
    print("\n2. Testing DMX API...")
    api_success = test_dmx_api()
    
    if jina_success and api_success:
        print("\n✅ All tests passed! SEMMA text semantics module is ready.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
